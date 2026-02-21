use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};

use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::{CudaMemoryClass, LoweredCudaPlan};
use crate::ir::{
    ExecutionPlan, Graph, PlacementClass, ShapeFact, ValueId, infer_shapes, plan_memory,
};

const MIN_BUFFER_ALIGNMENT_BYTES: usize = 16;
const DEVICE_BUFFER_ALIGNMENT_BYTES: usize = 256;

#[derive(Debug, Clone)]
pub struct CudaDeviceMemoryError {
    pub message: String,
}

#[derive(Debug)]
pub struct DeviceBuffer<T: DeviceRepr> {
    storage: CudaSlice<T>,
    len: usize,
    padded_len: usize,
    aligned_256: bool,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    pub fn alloc(device: &CudaDevice, len: usize) -> Result<Self, CudaDeviceMemoryError> {
        let stream = device.stream().map_err(|err| CudaDeviceMemoryError {
            message: err.message,
        })?;
        let padded_len = padded_len_for_allocation::<T>(len)?;
        let alloc_len = padded_len.max(1);
        let storage =
            unsafe { stream.alloc::<T>(alloc_len) }.map_err(|err| CudaDeviceMemoryError {
                message: format!("cudaMalloc failed for {} elements: {err}", padded_len),
            })?;
        let aligned_256 = pointer_is_256_aligned(&storage, &stream);

        Ok(Self {
            storage,
            len,
            padded_len,
            aligned_256,
        })
    }

    pub fn from_host(device: &CudaDevice, host: &[T]) -> Result<Self, CudaDeviceMemoryError> {
        let mut buffer = Self::alloc(device, host.len())?;
        buffer.copy_from_host(device, host)?;
        Ok(buffer)
    }

    pub fn copy_from_host(
        &mut self,
        device: &CudaDevice,
        host: &[T],
    ) -> Result<(), CudaDeviceMemoryError> {
        if host.is_empty() {
            return Ok(());
        }
        if host.len() > self.len {
            return Err(CudaDeviceMemoryError {
                message: format!(
                    "host copy overflow: host_len={} buffer_len={}",
                    host.len(),
                    self.len
                ),
            });
        }

        let stream = device.stream().map_err(|err| CudaDeviceMemoryError {
            message: err.message,
        })?;
        stream
            .memcpy_htod(host, &mut self.storage)
            .map_err(|err| CudaDeviceMemoryError {
                message: format!("cudaMemcpy H2D failed: {err}"),
            })?;
        Ok(())
    }

    pub fn copy_to_host(&self, device: &CudaDevice) -> Result<Vec<T>, CudaDeviceMemoryError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let stream = device.stream().map_err(|err| CudaDeviceMemoryError {
            message: err.message,
        })?;
        let mut host = stream
            .clone_dtoh(&self.storage)
            .map_err(|err| CudaDeviceMemoryError {
                message: format!("cudaMemcpy D2H failed: {err}"),
            })?;
        stream.synchronize().map_err(|err| CudaDeviceMemoryError {
            message: format!("CUDA stream synchronize failed after D2H copy: {err}"),
        })?;
        host.truncate(self.len);
        Ok(host)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    pub fn is_256_aligned(&self) -> bool {
        self.aligned_256
    }

    pub(crate) fn cuda_slice(&self) -> &CudaSlice<T> {
        &self.storage
    }

    pub(crate) fn cuda_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.storage
    }
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceBuffer<T> {
    pub fn zeros(device: &CudaDevice, len: usize) -> Result<Self, CudaDeviceMemoryError> {
        let stream = device.stream().map_err(|err| CudaDeviceMemoryError {
            message: err.message,
        })?;
        let padded_len = padded_len_for_allocation::<T>(len)?;
        let alloc_len = padded_len.max(1);
        let storage = stream
            .alloc_zeros::<T>(alloc_len)
            .map_err(|err| CudaDeviceMemoryError {
                message: format!(
                    "cudaMalloc zeroed failed for {} elements: {err}",
                    padded_len
                ),
            })?;
        let aligned_256 = pointer_is_256_aligned(&storage, &stream);

        Ok(Self {
            storage,
            len,
            padded_len,
            aligned_256,
        })
    }
}

fn padded_len_for_allocation<T: DeviceRepr>(len: usize) -> Result<usize, CudaDeviceMemoryError> {
    let elem_bytes = std::mem::size_of::<T>();
    if elem_bytes == 0 {
        return Err(CudaDeviceMemoryError {
            message: "zero-sized CUDA buffer element type is unsupported".to_string(),
        });
    }

    let total_bytes = len
        .checked_mul(elem_bytes)
        .ok_or_else(|| CudaDeviceMemoryError {
            message: format!("buffer byte size overflow: len={len} elem_bytes={elem_bytes}"),
        })?;
    if total_bytes == 0 {
        return Ok(0);
    }

    let padded_bytes =
        total_bytes.div_ceil(DEVICE_BUFFER_ALIGNMENT_BYTES) * DEVICE_BUFFER_ALIGNMENT_BYTES;
    Ok(padded_bytes.div_ceil(elem_bytes))
}

fn pointer_is_256_aligned<T: DeviceRepr>(
    storage: &CudaSlice<T>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> bool {
    let (ptr, _sync) = storage.device_ptr(stream);
    (ptr as usize).is_multiple_of(DEVICE_BUFFER_ALIGNMENT_BYTES)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryProfile {
    pub peak_device_bytes: usize,
    pub peak_host_transfer_bytes: usize,
    pub peak_temp_bytes: usize,
    pub alignment_consistent: bool,
    pub placement_fingerprint: u64,
}

#[derive(Debug, Clone)]
pub struct CudaMemoryProfileError {
    pub message: String,
}

pub fn profile_memory(
    graph: &Graph,
    plan: &ExecutionPlan,
    lowered: &LoweredCudaPlan,
) -> Result<CudaMemoryProfile, CudaMemoryProfileError> {
    verify_placement_mapping(plan, lowered)?;

    let memory_plan = plan_memory(graph).map_err(|err| CudaMemoryProfileError {
        message: err.message,
    })?;
    let shapes = infer_shapes(graph).map_err(|err| CudaMemoryProfileError {
        message: err.message,
    })?;

    let class_by_value = lowered
        .memory_bindings
        .iter()
        .map(|binding| (binding.value, binding.class))
        .collect::<HashMap<_, _>>();

    let peak_device_bytes = compute_peak_for_classes(
        &memory_plan.values,
        &plan.allocation.buffer_map,
        &class_by_value,
        |_| true,
    );
    let peak_temp_bytes = compute_peak_for_classes(
        &memory_plan.values,
        &plan.allocation.buffer_map,
        &class_by_value,
        |class| {
            matches!(
                class,
                CudaMemoryClass::Temporary | CudaMemoryClass::Gradient
            )
        },
    );

    let peak_host_transfer_bytes = lowered
        .memory_bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.class,
                CudaMemoryClass::Input | CudaMemoryClass::Parameter | CudaMemoryClass::Output
            )
        })
        .map(|binding| estimate_bytes(shapes.get(&binding.value)))
        .sum::<usize>();

    let alignment_consistent = memory_plan.values.iter().all(|value| {
        value.estimated_bytes == 0 || value.estimated_bytes % MIN_BUFFER_ALIGNMENT_BYTES == 0
    });

    let placement_fingerprint = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        lowered.memory_bindings.hash(&mut hasher);
        hasher.finish()
    };

    Ok(CudaMemoryProfile {
        peak_device_bytes,
        peak_host_transfer_bytes,
        peak_temp_bytes,
        alignment_consistent,
        placement_fingerprint,
    })
}

fn compute_peak_for_classes(
    liveness: &[crate::ir::ValueLiveness],
    buffer_map: &HashMap<ValueId, crate::ir::BufferId>,
    class_by_value: &HashMap<ValueId, CudaMemoryClass>,
    include: impl Fn(CudaMemoryClass) -> bool,
) -> usize {
    let max_point = liveness
        .iter()
        .map(|entry| entry.end_node)
        .max()
        .unwrap_or(0);
    let mut peak = 0usize;

    for point in 0..=max_point {
        let mut live_buffers = BTreeMap::<usize, usize>::new();
        for interval in liveness {
            if !(interval.start_node <= point && point <= interval.end_node) {
                continue;
            }

            let Some(class) = class_by_value.get(&interval.value).copied() else {
                continue;
            };
            if !include(class) {
                continue;
            }

            let Some(buffer) = buffer_map.get(&interval.value) else {
                continue;
            };
            live_buffers
                .entry(buffer.0)
                .or_insert(interval.estimated_bytes);
        }

        let total = live_buffers.values().copied().sum::<usize>();
        peak = peak.max(total);
    }

    peak
}

fn verify_placement_mapping(
    plan: &ExecutionPlan,
    lowered: &LoweredCudaPlan,
) -> Result<(), CudaMemoryProfileError> {
    if plan.placement_hints.len() != lowered.memory_bindings.len() {
        return Err(CudaMemoryProfileError {
            message: format!(
                "placement mapping mismatch: plan hints={} lowered bindings={}",
                plan.placement_hints.len(),
                lowered.memory_bindings.len()
            ),
        });
    }

    for (hint, binding) in plan
        .placement_hints
        .iter()
        .zip(lowered.memory_bindings.iter())
    {
        if hint.value != binding.value || map_placement(hint.class) != binding.class {
            return Err(CudaMemoryProfileError {
                message: format!("placement mapping mismatch for value {}", hint.value.0),
            });
        }
    }

    Ok(())
}

fn map_placement(class: PlacementClass) -> CudaMemoryClass {
    match class {
        PlacementClass::Input => CudaMemoryClass::Input,
        PlacementClass::Parameter => CudaMemoryClass::Parameter,
        PlacementClass::Temporary => CudaMemoryClass::Temporary,
        PlacementClass::Output => CudaMemoryClass::Output,
        PlacementClass::Gradient => CudaMemoryClass::Gradient,
    }
}

fn estimate_bytes(shape: Option<&ShapeFact>) -> usize {
    match shape {
        Some(ShapeFact::Tensor(dims)) => {
            if dims.contains(&0) {
                return 0;
            }

            let Some(raw_bytes) = dims
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            else {
                return 0;
            };

            raw_bytes
                .div_ceil(MIN_BUFFER_ALIGNMENT_BYTES)
                .saturating_mul(MIN_BUFFER_ALIGNMENT_BYTES)
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::cuda::{CudaMemoryClass, lower_plan, profile_memory};
    use crate::ir::{Graph, Op, build_execution_plan};

    #[test]
    fn profile_rejects_placement_mapping_mismatch() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add input should succeed");
        let (_, w) = graph
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add parameter should succeed");
        let (_, y) = graph
            .add_op(block, Op::MatMul(x, w))
            .expect("add matmul should succeed");
        graph
            .add_op(block, Op::Output(y))
            .expect("add output should succeed");
        graph.bind_input_shape("x", vec![1, 2]);
        graph.bind_parameter_shape("w", vec![2, 2]);

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
        let mut lowered = lower_plan(&plan).expect("lowering should pass");
        lowered.memory_bindings[0].class = CudaMemoryClass::Output;

        let err = profile_memory(&graph, &plan, &lowered)
            .expect_err("memory profile should reject placement mismatch");
        assert!(err.message.contains("placement mapping mismatch"));
    }
}

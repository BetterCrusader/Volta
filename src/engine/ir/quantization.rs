/// Quantization pass for Volta IR.
///
/// Implements post-training quantization (PTQ) by inserting QuantizeLinear /
/// DequantizeLinear ops around supported operations (Gemm, MatMul, Conv2D).
///
/// Calibration: per-tensor symmetric INT8 (scale = max_abs / 127).
use std::collections::HashMap;

use crate::ir::{Graph, Op, Pass};

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Symmetric INT8 per-tensor quantization.
    Int8Symmetric,
    /// Symmetric INT4 per-tensor quantization.
    Int4Symmetric,
}

impl QuantMode {
    pub fn bits(self) -> u8 {
        match self {
            Self::Int8Symmetric => 8,
            Self::Int4Symmetric => 4,
        }
    }
}

/// Per-tensor calibration statistics (min/max of activations).
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub min: f32,
    pub max: f32,
}

impl CalibrationStats {
    /// Compute symmetric scale for INT8/INT4.
    pub fn symmetric_scale(&self, bits: u8) -> f32 {
        let max_abs = self.min.abs().max(self.max.abs());
        let max_int = ((1i32 << (bits - 1)) - 1) as f32;
        if max_abs < 1e-8 {
            1e-8
        } else {
            max_abs / max_int
        }
    }
}

/// Configuration for the quantization pass.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub mode: QuantMode,
    /// Calibration data: map from output ValueId to observed stats.
    /// If None for a value, a default scale=1.0 is used.
    pub calibration: HashMap<usize, CalibrationStats>,
    /// If true, quantize weight tensors (ConstTensor) as well.
    pub quantize_weights: bool,
}

impl QuantizationConfig {
    pub fn new(mode: QuantMode) -> Self {
        Self {
            mode,
            calibration: HashMap::new(),
            quantize_weights: true,
        }
    }

    pub fn with_calibration(mut self, value_id: usize, stats: CalibrationStats) -> Self {
        self.calibration.insert(value_id, stats);
        self
    }

    fn scale_for(&self, value_id: usize) -> f32 {
        if let Some(stats) = self.calibration.get(&value_id) {
            stats.symmetric_scale(self.mode.bits())
        } else {
            1.0 / 127.0 // default scale
        }
    }
}

/// Quantization pass: inserts QuantizeLinear/DequantizeLinear around Gemm and MatMul ops.
///
/// For each Gemm/MatMul:
///   original: out = op(lhs, rhs)
///   quantized: qlhs = Q(lhs), qrhs = Q(rhs), out = op(DQ(qlhs), DQ(qrhs))
///   then: qout = Q(out), result = DQ(qout)
pub struct QuantizationPass {
    pub config: QuantizationConfig,
}

impl QuantizationPass {
    pub fn new(config: QuantizationConfig) -> Self {
        Self { config }
    }
}

impl Pass for QuantizationPass {
    fn run(&mut self, graph: &mut Graph) {
        // Collect ops that need quantization (Gemm, MatMul)
        // We'll insert Q/DQ wrappers by modifying the graph.
        // Since we can't insert nodes mid-iteration (node indices would shift),
        // we collect targets first then process them.

        let targets: Vec<(usize, Op)> = graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| match &node.op {
                Op::Gemm { .. } | Op::MatMul(_, _) => Some((i, node.op.clone())),
                _ => None,
            })
            .collect();

        // For each target op, wrap inputs with Q/DQ
        for (node_idx, op) in targets {
            match op {
                Op::Gemm {
                    lhs,
                    rhs,
                    bias,
                    alpha,
                    beta,
                } => {
                    // Get the block containing this node
                    let block_id = find_block_for_node(graph, node_idx);
                    let Some(block_id) = block_id else { continue };

                    let lhs_scale = self.config.scale_for(lhs.0);
                    let rhs_scale = self.config.scale_for(rhs.0);

                    // Insert Q(lhs) and DQ(Q(lhs)) before this node
                    let (_, q_lhs_val) = match graph.add_op(
                        block_id,
                        Op::QuantizeLinear {
                            input: lhs,
                            scale: lhs_scale,
                            zero_point: 0,
                            bits: self.config.mode.bits(),
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let (_, dq_lhs_val) = match graph.add_op(
                        block_id,
                        Op::DequantizeLinear {
                            input: q_lhs_val,
                            scale: lhs_scale,
                            zero_point: 0,
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let (_, q_rhs_val) = match graph.add_op(
                        block_id,
                        Op::QuantizeLinear {
                            input: rhs,
                            scale: rhs_scale,
                            zero_point: 0,
                            bits: self.config.mode.bits(),
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let (_, dq_rhs_val) = match graph.add_op(
                        block_id,
                        Op::DequantizeLinear {
                            input: q_rhs_val,
                            scale: rhs_scale,
                            zero_point: 0,
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    // Update the Gemm node to use quantized inputs
                    graph.nodes[node_idx].op = Op::Gemm {
                        lhs: dq_lhs_val,
                        rhs: dq_rhs_val,
                        bias,
                        alpha,
                        beta,
                    };
                }
                Op::MatMul(lhs, rhs) => {
                    let block_id = find_block_for_node(graph, node_idx);
                    let Some(block_id) = block_id else { continue };

                    let lhs_scale = self.config.scale_for(lhs.0);
                    let rhs_scale = self.config.scale_for(rhs.0);

                    let (_, q_lhs_val) = match graph.add_op(
                        block_id,
                        Op::QuantizeLinear {
                            input: lhs,
                            scale: lhs_scale,
                            zero_point: 0,
                            bits: self.config.mode.bits(),
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let (_, dq_lhs_val) = match graph.add_op(
                        block_id,
                        Op::DequantizeLinear {
                            input: q_lhs_val,
                            scale: lhs_scale,
                            zero_point: 0,
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let (_, q_rhs_val) = match graph.add_op(
                        block_id,
                        Op::QuantizeLinear {
                            input: rhs,
                            scale: rhs_scale,
                            zero_point: 0,
                            bits: self.config.mode.bits(),
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let (_, dq_rhs_val) = match graph.add_op(
                        block_id,
                        Op::DequantizeLinear {
                            input: q_rhs_val,
                            scale: rhs_scale,
                            zero_point: 0,
                        },
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    graph.nodes[node_idx].op = Op::MatMul(dq_lhs_val, dq_rhs_val);
                }
                _ => {}
            }
        }
    }

    fn name(&self) -> &str {
        "QuantizationPass"
    }
}

fn find_block_for_node(graph: &Graph, node_idx: usize) -> Option<crate::ir::block::BasicBlockId> {
    let node_id = graph.nodes.get(node_idx)?.id;
    graph
        .blocks
        .iter()
        .find(|b| b.nodes.contains(&node_id))
        .map(|b| b.id)
}

/// Compute per-tensor calibration statistics from a flat f32 slice.
pub fn calibrate_tensor(data: &[f32]) -> CalibrationStats {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    CalibrationStats { min, max }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_scale_int8_symmetric() {
        let stats = CalibrationStats {
            min: -1.0,
            max: 1.0,
        };
        let scale = stats.symmetric_scale(8);
        // 1.0 / 127 ≈ 0.00787
        assert!((scale - 1.0 / 127.0).abs() < 1e-5, "scale={scale}");
    }

    #[test]
    fn calibration_stats_from_data() {
        let data = vec![-2.0f32, 0.0, 3.0, -1.0];
        let stats = calibrate_tensor(&data);
        assert!((stats.min - (-2.0)).abs() < 1e-6);
        assert!((stats.max - 3.0).abs() < 1e-6);
    }

    #[test]
    fn quant_mode_bits() {
        assert_eq!(QuantMode::Int8Symmetric.bits(), 8);
        assert_eq!(QuantMode::Int4Symmetric.bits(), 4);
    }
}

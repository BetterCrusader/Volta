use crate::ir::{Graph, Op, ValueId};

/// Returns a short human-readable name for an op variant (for profiler display).
pub fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Add(..) => "Add",
        Op::Sub(..) => "Sub",
        Op::Mul(..) => "Mul",
        Op::Div(..) => "Div",
        Op::MatMul(..) => "MatMul",
        Op::Gemm { .. } => "Gemm",
        Op::GemmBackward { .. } => "GemmBackward",
        Op::Relu(..) => "Relu",
        Op::Sigmoid(..) => "Sigmoid",
        Op::Gelu(..) => "Gelu",
        Op::GeluExact(..) => "GeluExact",
        Op::Softmax(..) => "Softmax",
        Op::LayerNorm { .. } => "LayerNorm",
        Op::BatchNorm { .. } => "BatchNorm",
        Op::GroupNorm { .. } => "GroupNorm",
        Op::InstanceNorm { .. } => "InstanceNorm",
        Op::Dropout { .. } => "Dropout",
        Op::Flatten { .. } => "Flatten",
        Op::GlobalAveragePool { .. } => "GlobalAveragePool",
        Op::GlobalAveragePoolBackward { .. } => "GlobalAveragePoolBackward",
        Op::Conv2D(..) => "Conv2D",
        Op::Conv2DBackwardInput(..) => "Conv2DBackwardInput",
        Op::Conv2DBackwardWeight(..) => "Conv2DBackwardWeight",
        Op::ConvTranspose2D { .. } => "ConvTranspose2D",
        Op::MaxPool { .. } => "MaxPool",
        Op::AvgPool { .. } => "AvgPool",
        Op::MaxPoolBackward { .. } => "MaxPoolBackward",
        Op::AvgPoolBackward { .. } => "AvgPoolBackward",
        Op::Embedding { .. } => "Embedding",
        Op::EmbeddingBackward { .. } => "EmbeddingBackward",
        Op::LstmCell { .. } => "LstmCell",
        Op::LstmCellBackward { .. } => "LstmCellBackward",
        Op::GruCell { .. } => "GruCell",
        Op::GruCellBackward { .. } => "GruCellBackward",
        Op::MultiHeadAttention { .. } => "MultiHeadAttention",
        Op::MultiHeadAttentionBackward { .. } => "MultiHeadAttentionBackward",
        Op::SinusoidalPE { .. } => "SinusoidalPE",
        Op::RoPE { .. } => "RoPE",
        Op::RoPEBackward { .. } => "RoPEBackward",
        Op::Upsample2D { .. } => "Upsample2D",
        Op::Upsample2DBackward { .. } => "Upsample2DBackward",
        Op::ReduceSum { .. } => "ReduceSum",
        Op::ReduceMax { .. } => "ReduceMax",
        Op::ReduceMean { .. } => "ReduceMean",
        Op::Reshape { .. } => "Reshape",
        Op::Transpose(..) => "Transpose",
        Op::Concat { .. } => "Concat",
        Op::Gather { .. } => "Gather",
        Op::Slice { .. } => "Slice",
        Op::Parameter(..) => "Parameter",
        Op::Input(..) => "Input",
        Op::Output(..) => "Output",
        Op::ConstInt(..) => "ConstInt",
        Op::ConstFloat(..) => "ConstFloat",
        Op::ConstTensor { .. } => "ConstTensor",
        Op::Identity(..) => "Identity",
        Op::Phi(..) => "Phi",
        Op::Plugin { .. } => "Plugin",
        Op::CustomCall { .. } => "CustomCall",
        Op::DepthwiseSeparableConv { .. } => "DepthwiseSeparableConv",
        Op::QuantizeLinear { .. } => "QuantizeLinear",
        Op::DequantizeLinear { .. } => "DequantizeLinear",
        Op::Removed => "Removed",
        _ => "Unknown",
    }
}

#[must_use]
pub fn print_graph(graph: &Graph) -> String {
    let mut lines = Vec::new();
    for node in &graph.nodes {
        lines.push(format!("%{} = {}", node.output.0, format_op(&node.op)));
    }
    lines.join("\n")
}

fn format_op(op: &Op) -> String {
    match op {
        Op::ConstInt(value) => format!("const {value}"),
        Op::ConstFloat(value) => format!("const {value}"),
        Op::ConstTensor { shape, .. } => format!("const_tensor {shape:?}"),
        Op::Add(left, right) => format!("add {} {}", fmt_value(left), fmt_value(right)),
        Op::Sub(left, right) => format!("sub {} {}", fmt_value(left), fmt_value(right)),
        Op::Mul(left, right) => format!("mul {} {}", fmt_value(left), fmt_value(right)),
        Op::Div(left, right) => format!("div {} {}", fmt_value(left), fmt_value(right)),
        Op::Neg(value) => format!("neg {}", fmt_value(value)),
        Op::ElementwiseChain { input, ops } => {
            let chain = ops
                .iter()
                .map(|op| match op {
                    crate::ir::ElementwiseUnaryOp::Neg => "neg",
                    crate::ir::ElementwiseUnaryOp::Relu => "relu",
                    crate::ir::ElementwiseUnaryOp::Sigmoid => "sigmoid",
                    crate::ir::ElementwiseUnaryOp::Gelu => "gelu",
                    crate::ir::ElementwiseUnaryOp::GeluExact => "gelu_exact",
                    crate::ir::ElementwiseUnaryOp::Exp => "exp",
                    crate::ir::ElementwiseUnaryOp::Log => "log",
                    crate::ir::ElementwiseUnaryOp::LeakyRelu(_) => "leaky_relu",
                })
                .collect::<Vec<_>>()
                .join("->");
            format!("elementwise_chain {} [{}]", fmt_value(input), chain)
        }
        Op::Reshape { input, shape } => format!("reshape {} {:?}", fmt_value(input), shape),
        Op::Concat { inputs, axis } => {
            let formatted = inputs
                .iter()
                .map(|id| fmt_value(id))
                .collect::<Vec<_>>()
                .join(" ");
            format!("concat axis={axis} {formatted}")
        }
        Op::Gather {
            input,
            indices,
            axis,
        } => format!("gather axis={} {} {:?}", axis, fmt_value(input), indices),
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => format!(
            "slice {} starts={:?} ends={:?} axes={:?}",
            fmt_value(input),
            starts,
            ends,
            axes
        ),
        Op::Transpose(value) => format!("transpose {}", fmt_value(value)),
        Op::MatMul(left, right) => format!("matmul {} {}", fmt_value(left), fmt_value(right)),
        Op::Relu(value) => format!("relu {}", fmt_value(value)),
        Op::ReluBackward(input, grad) => {
            format!("relu_backward {} {}", fmt_value(input), fmt_value(grad))
        }
        Op::Softmax(value) => format!("softmax {}", fmt_value(value)),
        Op::Log(value) => format!("log {}", fmt_value(value)),
        Op::Exp(value) => format!("exp {}", fmt_value(value)),
        Op::Sigmoid(value) => format!("sigmoid {}", fmt_value(value)),
        Op::SigmoidBackward(input, grad) => {
            format!("sigmoid_backward {} {}", fmt_value(input), fmt_value(grad))
        }
        Op::Gelu(value) => format!("gelu {}", fmt_value(value)),
        Op::GeluExact(value) => format!("gelu_exact {}", fmt_value(value)),
        Op::GeluBackward(input, grad) => {
            format!("gelu_backward {} {}", fmt_value(input), fmt_value(grad))
        }
        Op::GeluExactBackward(input, grad) => {
            format!(
                "gelu_exact_backward {} {}",
                fmt_value(input),
                fmt_value(grad)
            )
        }
        Op::ReduceMaxBackward {
            input,
            output_max,
            upstream,
            axis,
            keepdims,
        } => {
            format!(
                "reduce_max_backward {} {} {} axis={:?} keepdims={}",
                fmt_value(input),
                fmt_value(output_max),
                fmt_value(upstream),
                axis,
                keepdims
            )
        }
        Op::Gemm {
            lhs,
            rhs,
            bias,
            alpha,
            beta,
        } => {
            let bias_str = bias
                .map(|b| format!(" {}", fmt_value(&b)))
                .unwrap_or_default();
            format!(
                "gemm {} {}{} alpha={} beta={}",
                fmt_value(lhs),
                fmt_value(rhs),
                bias_str,
                alpha,
                beta
            )
        }
        Op::GemmBackward {
            lhs,
            rhs,
            bias,
            alpha,
            beta,
        } => {
            let bias_str = bias
                .map(|b| format!(" {}", fmt_value(&b)))
                .unwrap_or_default();
            format!(
                "gemm_backward {} {}{} alpha={} beta={}",
                fmt_value(lhs),
                fmt_value(rhs),
                bias_str,
                alpha,
                beta
            )
        }
        Op::Plugin { operator, inputs } => {
            let inputs_str = inputs
                .iter()
                .map(|id| fmt_value(id))
                .collect::<Vec<_>>()
                .join(" ");
            format!("plugin {} {}", operator.name(), inputs_str)
        }
        Op::CustomCall { target, inputs, .. } => {
            let inputs_str = inputs
                .iter()
                .map(|id| fmt_value(id))
                .collect::<Vec<_>>()
                .join(" ");
            format!("custom_call target={target} {inputs_str}")
        }
        Op::DepthwiseSeparableConv {
            input,
            dw_weight,
            pw_weight,
            stride,
            padding,
        } => {
            format!(
                "depthwise_sep_conv stride={},{} pad={},{} {} {} {}",
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                fmt_value(input),
                fmt_value(dw_weight),
                fmt_value(pw_weight)
            )
        }
        Op::QuantizeLinear {
            input,
            scale,
            zero_point,
            bits,
        } => {
            format!(
                "quantize_linear scale={scale} zp={zero_point} bits={bits} {}",
                fmt_value(input)
            )
        }
        Op::DequantizeLinear {
            input,
            scale,
            zero_point,
        } => {
            format!(
                "dequantize_linear scale={scale} zp={zero_point} {}",
                fmt_value(input)
            )
        }
        Op::ReduceSum {
            input,
            axis,
            keepdims,
        } => match (axis, keepdims) {
            (Some(a), true) => format!("reduce_sum axis={} keepdims=1 {}", a, fmt_value(input)),
            (Some(a), false) => format!("reduce_sum axis={} {}", a, fmt_value(input)),
            (None, true) => format!("reduce_sum keepdims=1 {}", fmt_value(input)),
            (None, false) => format!("reduce_sum {}", fmt_value(input)),
        },
        Op::ReduceMax {
            input,
            axis,
            keepdims,
        } => match (axis, keepdims) {
            (Some(a), true) => format!("reduce_max axis={} keepdims=1 {}", a, fmt_value(input)),
            (Some(a), false) => format!("reduce_max axis={} {}", a, fmt_value(input)),
            (None, true) => format!("reduce_max keepdims=1 {}", fmt_value(input)),
            (None, false) => format!("reduce_max {}", fmt_value(input)),
        },
        Op::ReduceMean {
            input,
            axis,
            keepdims,
        } => match (axis, keepdims) {
            (Some(a), true) => {
                format!("reduce_mean axis={} keepdims=1 {}", a, fmt_value(input))
            }
            (Some(a), false) => format!("reduce_mean axis={} {}", a, fmt_value(input)),
            (None, true) => format!("reduce_mean keepdims=1 {}", fmt_value(input)),
            (None, false) => format!("reduce_mean {}", fmt_value(input)),
        },
        Op::Conv2D(input, weight) => {
            format!("conv2d {} {}", fmt_value(input), fmt_value(weight))
        }
        Op::Conv2DBackwardInput(input, weight, upstream) => {
            format!(
                "conv2d_backward_input {} {} {}",
                fmt_value(input),
                fmt_value(weight),
                fmt_value(upstream)
            )
        }
        Op::Conv2DBackwardWeight(input, weight, upstream) => {
            format!(
                "conv2d_backward_weight {} {} {}",
                fmt_value(input),
                fmt_value(weight),
                fmt_value(upstream)
            )
        }
        Op::MaxPool {
            input,
            kernel_shape,
            strides,
            pads,
        } => format!(
            "max_pool {} kernel={:?} strides={:?} pads={:?}",
            fmt_value(input),
            kernel_shape,
            strides,
            pads
        ),
        Op::AvgPool {
            input,
            kernel_shape,
            strides,
            pads,
        } => format!(
            "avg_pool {} kernel={:?} strides={:?} pads={:?}",
            fmt_value(input),
            kernel_shape,
            strides,
            pads
        ),
        Op::BatchNorm {
            input,
            weight,
            bias,
            mean,
            var,
        } => format!(
            "batch_norm {} {} {} {} {}",
            fmt_value(input),
            fmt_value(weight),
            fmt_value(bias),
            fmt_value(mean),
            fmt_value(var)
        ),
        Op::Flatten { input, axis } => format!("flatten {} axis={}", fmt_value(input), axis),
        Op::GlobalAveragePool { input } => format!("global_avg_pool {}", fmt_value(input)),
        Op::GlobalAveragePoolBackward { input, upstream } => {
            format!(
                "global_avg_pool_bwd {} upstream={}",
                fmt_value(input),
                fmt_value(upstream)
            )
        }
        Op::GroupNorm {
            input,
            weight,
            bias,
            num_groups,
            ..
        } => {
            format!(
                "group_norm {} weight={} bias={} groups={}",
                fmt_value(input),
                fmt_value(weight),
                fmt_value(bias),
                num_groups
            )
        }
        Op::GroupNormBackwardInput {
            input,
            upstream,
            num_groups,
            ..
        } => {
            format!(
                "group_norm_bwd_input {} upstream={} groups={}",
                fmt_value(input),
                fmt_value(upstream),
                num_groups
            )
        }
        Op::GroupNormBackwardWeight {
            input,
            upstream,
            num_groups,
            ..
        } => {
            format!(
                "group_norm_bwd_weight {} upstream={} groups={}",
                fmt_value(input),
                fmt_value(upstream),
                num_groups
            )
        }
        Op::GroupNormBackwardBias { upstream } => {
            format!("group_norm_bwd_bias {}", fmt_value(upstream))
        }
        Op::InstanceNorm {
            input,
            weight,
            bias,
            ..
        } => {
            format!(
                "instance_norm {} weight={} bias={}",
                fmt_value(input),
                fmt_value(weight),
                fmt_value(bias)
            )
        }
        Op::InstanceNormBackwardInput {
            input, upstream, ..
        } => {
            format!(
                "instance_norm_bwd_input {} upstream={}",
                fmt_value(input),
                fmt_value(upstream)
            )
        }
        Op::InstanceNormBackwardWeight {
            input, upstream, ..
        } => {
            format!(
                "instance_norm_bwd_weight {} upstream={}",
                fmt_value(input),
                fmt_value(upstream)
            )
        }
        Op::InstanceNormBackwardBias { upstream } => {
            format!("instance_norm_bwd_bias {}", fmt_value(upstream))
        }
        Op::Embedding { weight, indices } => {
            format!(
                "embedding weight={} indices={}",
                fmt_value(weight),
                fmt_value(indices)
            )
        }
        Op::EmbeddingBackward {
            weight,
            indices,
            upstream,
        } => {
            format!(
                "embedding_bwd weight={} indices={} upstream={}",
                fmt_value(weight),
                fmt_value(indices),
                fmt_value(upstream)
            )
        }
        Op::LstmCell {
            x,
            h_prev,
            output_idx,
            ..
        } => {
            format!(
                "lstm_cell x={} h={} out={}",
                fmt_value(x),
                fmt_value(h_prev),
                output_idx
            )
        }
        Op::LstmCellBackward { x, grad_target, .. } => {
            format!("lstm_cell_bwd x={} grad={}", fmt_value(x), grad_target)
        }
        Op::GruCell {
            x,
            h_prev,
            output_idx,
            ..
        } => {
            format!(
                "gru_cell x={} h={} out={}",
                fmt_value(x),
                fmt_value(h_prev),
                output_idx
            )
        }
        Op::GruCellBackward { x, grad_target, .. } => {
            format!("gru_cell_bwd x={} grad={}", fmt_value(x), grad_target)
        }
        Op::ConvTranspose2D {
            input,
            weight,
            stride,
            padding,
        } => {
            format!(
                "conv_transpose2d input={} weight={} stride={:?} pad={:?}",
                fmt_value(input),
                fmt_value(weight),
                stride,
                padding
            )
        }
        Op::Upsample2D {
            input,
            scale_h,
            scale_w,
            mode,
        } => {
            format!(
                "upsample2d {} scale=({}, {}) mode={}",
                fmt_value(input),
                scale_h,
                scale_w,
                if *mode == 0 { "nearest" } else { "bilinear" }
            )
        }
        Op::Upsample2DBackward {
            upstream,
            orig_h,
            orig_w,
            ..
        } => {
            format!(
                "upsample2d_bwd {} orig=({}, {})",
                fmt_value(upstream),
                orig_h,
                orig_w
            )
        }
        Op::MultiHeadAttention {
            q_input,
            num_heads,
            output_idx,
            causal,
            ..
        } => {
            format!(
                "mha q={} heads={} out={} causal={}",
                fmt_value(q_input),
                num_heads,
                output_idx,
                causal
            )
        }
        Op::SinusoidalPE { input } => format!("sinusoidal_pe {}", fmt_value(input)),
        Op::RoPE { input, offset } => format!("rope {} offset={}", fmt_value(input), offset),
        Op::RoPEBackward { upstream, offset } => {
            format!("rope_bwd {} offset={}", fmt_value(upstream), offset)
        }
        Op::Dropout { input, ratio } => {
            format!("dropout {} ratio={}", fmt_value(input), ratio)
        }
        Op::Identity(value) => format!("identity {}", fmt_value(value)),
        Op::MaxPoolBackward {
            input,
            upstream,
            kernel_shape,
            strides,
            pads,
        } => format!(
            "max_pool_backward {} {} kernel={:?} strides={:?} pads={:?}",
            fmt_value(input),
            fmt_value(upstream),
            kernel_shape,
            strides,
            pads
        ),
        Op::AvgPoolBackward {
            input,
            upstream,
            kernel_shape,
            strides,
            pads,
        } => format!(
            "avg_pool_backward {} {} kernel={:?} strides={:?} pads={:?}",
            fmt_value(input),
            fmt_value(upstream),
            kernel_shape,
            strides,
            pads
        ),
        Op::BatchNormBackwardInput {
            input,
            upstream,
            weight,
            var,
        } => format!(
            "batch_norm_backward_input {} {} {} {}",
            fmt_value(input),
            fmt_value(upstream),
            fmt_value(weight),
            fmt_value(var)
        ),
        Op::BatchNormBackwardWeight {
            input,
            upstream,
            mean,
            var,
        } => format!(
            "batch_norm_backward_weight {} {} {} {}",
            fmt_value(input),
            fmt_value(upstream),
            fmt_value(mean),
            fmt_value(var)
        ),
        Op::BatchNormBackwardBias { upstream } => {
            format!("batch_norm_backward_bias {}", fmt_value(upstream))
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            epsilon,
        } => format!(
            "layer_norm {} {} {} eps={}",
            fmt_value(input),
            fmt_value(weight),
            fmt_value(bias),
            epsilon
        ),
        Op::LayerNormBackwardInput {
            input,
            upstream,
            weight,
            epsilon,
        } => format!(
            "layer_norm_backward_input {} {} {} eps={}",
            fmt_value(input),
            fmt_value(upstream),
            fmt_value(weight),
            epsilon
        ),
        Op::LayerNormBackwardWeight {
            input,
            upstream,
            epsilon,
        } => format!(
            "layer_norm_backward_weight {} {} eps={}",
            fmt_value(input),
            fmt_value(upstream),
            epsilon
        ),
        Op::LayerNormBackwardBias { upstream } => {
            format!("layer_norm_backward_bias {}", fmt_value(upstream))
        }
        Op::Parameter(name) => format!("parameter {name}"),
        Op::Input(name) => format!("input {name}"),
        Op::Output(value) => format!("output {}", fmt_value(value)),
        Op::Phi(values) => {
            let formatted = values
                .iter()
                .map(|id| fmt_value(id))
                .collect::<Vec<_>>()
                .join(" ");
            format!("phi {formatted}")
        }
        Op::Removed => "removed".to_string(),
        Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => {
            format!(
                "softmax_cross_entropy {} {}",
                fmt_value(logits),
                fmt_value(targets)
            )
        }
        Op::MultiHeadAttentionBackward {
            q_input,
            bias_q,
            bias_k,
            bias_v,
            num_heads,
            output_idx,
            ..
        } => {
            format!(
                "mha_bwd q={} bq={} bk={} bv={} heads={} out={}",
                fmt_value(q_input),
                fmt_value(bias_q),
                fmt_value(bias_k),
                fmt_value(bias_v),
                num_heads,
                output_idx
            )
        }
    }
}

fn fmt_value(value: &ValueId) -> String {
    format!("%{}", value.0)
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op};

    use super::print_graph;

    #[test]
    fn prints_ssa_style_lines() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(5))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let actual = print_graph(&graph);
        let expected = "%0 = const 5\n%1 = const 3\n%2 = add %0 %1";
        assert_eq!(actual, expected);
    }

    #[test]
    fn snapshot_print_for_fused_tensor_chain() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![1.0, -2.0],
                },
            )
            .expect("add op should succeed");
        let (_, relu) = graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Neg(relu))
            .expect("add op should succeed");

        let actual = print_graph(&graph);
        let expected = "%0 = const_tensor [2]\n%1 = relu %0\n%2 = neg %1";
        assert_eq!(actual, expected);
    }
}

use crate::ir::node::ValueId;

#[derive(Debug, Clone)]
pub enum ElementwiseUnaryOp {
    LeakyRelu(f32),
    Neg,
    Relu,
    Sigmoid,
    Gelu,
    GeluExact,
    Exp,
    Log,
}

impl PartialEq for ElementwiseUnaryOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::LeakyRelu(a), Self::LeakyRelu(b)) => a.to_bits() == b.to_bits(),
            (Self::Neg, Self::Neg) => true,
            (Self::Relu, Self::Relu) => true,
            (Self::Sigmoid, Self::Sigmoid) => true,
            (Self::Gelu, Self::Gelu) => true,
            (Self::GeluExact, Self::GeluExact) => true,
            (Self::Exp, Self::Exp) => true,
            (Self::Log, Self::Log) => true,
            _ => false,
        }
    }
}

impl Eq for ElementwiseUnaryOp {}

impl std::hash::Hash for ElementwiseUnaryOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::LeakyRelu(a) => {
                0_u8.hash(state);
                a.to_bits().hash(state);
            }
            Self::Neg => 1_u8.hash(state),
            Self::Relu => 2_u8.hash(state),
            Self::Sigmoid => 3_u8.hash(state),
            Self::Gelu => 4_u8.hash(state),
            Self::GeluExact => 5_u8.hash(state),
            Self::Exp => 6_u8.hash(state),
            Self::Log => 7_u8.hash(state),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Op {
    ConstInt(i64),
    ConstFloat(f64),
    ConstTensor {
        shape: Vec<usize>,
        data: Vec<f32>,
    },
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Neg(ValueId),
    ElementwiseChain {
        input: ValueId,
        ops: Vec<ElementwiseUnaryOp>,
    },
    Reshape {
        input: ValueId,
        shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<ValueId>,
        axis: usize,
    },
    Gather {
        input: ValueId,
        indices: Vec<usize>,
        axis: usize,
    },
    Slice {
        input: ValueId,
        starts: Vec<usize>,
        ends: Vec<usize>,
        axes: Vec<usize>,
    },
    Transpose(ValueId),
    MatMul(ValueId, ValueId),
    Relu(ValueId),
    ReluBackward(ValueId, ValueId),
    Softmax(ValueId),
    Log(ValueId),
    Exp(ValueId),
    Sigmoid(ValueId),
    SigmoidBackward(ValueId, ValueId),
    Gelu(ValueId),
    GeluExact(ValueId),
    GeluBackward(ValueId, ValueId),
    GeluExactBackward(ValueId, ValueId),
    SoftmaxCrossEntropyLossFromLogits {
        logits: ValueId,
        targets: ValueId,
    },
    ReduceSum {
        input: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMax {
        input: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMaxBackward {
        input: ValueId,
        output_max: ValueId,
        upstream: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMean {
        input: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    /// Generalised Matrix Multiply: `alpha * (lhs × rhs) + beta * bias`.
    ///
    /// Matches the ONNX `Gemm` operator and the standard BLAS SGEMM interface.
    /// When `bias` is `None` the operation degenerates to a scaled MatMul.
    Gemm {
        lhs: ValueId,
        rhs: ValueId,
        bias: Option<ValueId>,
        alpha: f32,
        beta: f32,
    },
    GemmBackward {
        lhs: ValueId,
        rhs: ValueId,
        bias: Option<ValueId>,
        alpha: f32,
        beta: f32,
    },
    Conv2D(ValueId, ValueId),
    Conv2DBackwardInput(ValueId, ValueId, ValueId),
    Conv2DBackwardWeight(ValueId, ValueId, ValueId),
    MaxPool {
        input: ValueId,
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    AvgPool {
        input: ValueId,
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    BatchNorm {
        input: ValueId,
        weight: ValueId,
        bias: ValueId,
        mean: ValueId,
        var: ValueId,
    },
    LayerNorm {
        input: ValueId,
        weight: ValueId,
        bias: ValueId,
        epsilon: f32,
    },
    Flatten {
        input: ValueId,
        axis: usize,
    },
    GlobalAveragePool {
        input: ValueId,
    },
    GlobalAveragePoolBackward {
        input: ValueId,
        upstream: ValueId,
    },
    GroupNorm {
        input: ValueId,
        weight: ValueId,
        bias: ValueId,
        num_groups: usize,
        epsilon: f32,
    },
    GroupNormBackwardInput {
        input: ValueId,
        upstream: ValueId,
        weight: ValueId,
        num_groups: usize,
        epsilon: f32,
    },
    GroupNormBackwardWeight {
        input: ValueId,
        upstream: ValueId,
        num_groups: usize,
        epsilon: f32,
    },
    GroupNormBackwardBias {
        upstream: ValueId,
    },
    InstanceNorm {
        input: ValueId,
        weight: ValueId,
        bias: ValueId,
        epsilon: f32,
    },
    InstanceNormBackwardInput {
        input: ValueId,
        upstream: ValueId,
        weight: ValueId,
        epsilon: f32,
    },
    InstanceNormBackwardWeight {
        input: ValueId,
        upstream: ValueId,
        epsilon: f32,
    },
    InstanceNormBackwardBias {
        upstream: ValueId,
    },
    /// Embedding lookup: weight [vocab_size, embed_dim], indices (as f32 integers) [batch*, seq_len] or [seq_len].
    /// Output: [..., embed_dim].
    Embedding {
        weight: ValueId,
        indices: ValueId,
    },
    /// Sparse gradient for the embedding weight: same shape as weight [vocab_size, embed_dim].
    /// Accumulates upstream gradients at the looked-up index positions.
    EmbeddingBackward {
        weight: ValueId,
        indices: ValueId,
        upstream: ValueId,
    },
    /// LSTM cell forward. Outputs h_next; c_next, gates_raw, tanh_c_next saved for backward.
    /// Encodes output selection: 0=h_next, 1=c_next, 2=gates_raw, 3=tanh_c_next.
    LstmCell {
        x: ValueId,
        h_prev: ValueId,
        c_prev: ValueId,
        weight_ih: ValueId,
        weight_hh: ValueId,
        bias: ValueId,
        /// Which output to expose: 0=h_next, 1=c_next, 2=gates_raw, 3=tanh_c_next
        output_idx: usize,
    },
    /// LSTM cell backward — grad for one of the inputs.
    /// grad_target: 0=dx, 1=dh_prev, 2=dc_prev, 3=dweight_ih, 4=dweight_hh, 5=dbias
    LstmCellBackward {
        x: ValueId,
        h_prev: ValueId,
        c_prev: ValueId,
        weight_ih: ValueId,
        weight_hh: ValueId,
        gates_raw: ValueId,
        tanh_c_next: ValueId,
        dh_next: ValueId,
        dc_next: ValueId,
        grad_target: usize,
    },
    /// GRU cell forward. output_idx: 0=h_next, 1=z_gate, 2=r_gate, 3=n_gate.
    GruCell {
        x: ValueId,
        h_prev: ValueId,
        weight_ih: ValueId,
        weight_hh: ValueId,
        bias_ih: ValueId,
        bias_hh: ValueId,
        output_idx: usize,
    },
    /// GRU cell backward. grad_target: 0=dx, 1=dh_prev, 2=dweight_ih, 3=dweight_hh, 4=dbias_ih, 5=dbias_hh.
    GruCellBackward {
        x: ValueId,
        h_prev: ValueId,
        weight_ih: ValueId,
        weight_hh: ValueId,
        z_gate: ValueId,
        r_gate: ValueId,
        n_gate: ValueId,
        dh_next: ValueId,
        grad_target: usize,
    },
    /// Sinusoidal positional encoding: input [batch, seq_len, d_model] or [seq_len, d_model].
    /// Returns input + sinusoidal PE (gradient passes through unchanged).
    SinusoidalPE {
        input: ValueId,
    },
    /// Rotary Position Embedding (RoPE): rotate last dim in pairs.
    /// input [batch, seq_len, head_dim] or [batch, heads, seq_len, head_dim].
    RoPE {
        input: ValueId,
        /// Starting sequence position offset (for KV cache use)
        offset: usize,
    },
    /// Transposed convolution (deconvolution) for generative models.
    /// input: [N, C_in, H, W], weight: [C_in, C_out, kH, kW]
    ConvTranspose2D {
        input: ValueId,
        weight: ValueId,
        stride: [usize; 2],
        padding: [usize; 2],
    },
    /// Upsample 2D. mode: 0=nearest, 1=bilinear.
    /// scale_h, scale_w stored as f32 (nearest uses round to usize).
    Upsample2D {
        input: ValueId,
        scale_h: f32,
        scale_w: f32,
        /// 0 = nearest, 1 = bilinear
        mode: usize,
    },
    /// Upsample backward (nearest only for now).
    Upsample2DBackward {
        upstream: ValueId,
        orig_h: usize,
        orig_w: usize,
        scale_h: usize,
        scale_w: usize,
    },
    /// RoPE backward: applies inverse rotation to upstream gradient.
    RoPEBackward {
        upstream: ValueId,
        offset: usize,
    },
    /// Multi-Head Attention forward.
    /// q_input, k_input, v_input: [batch, seq, d_model]
    /// w_q, w_k, w_v, w_o: [d_model, d_model] (weight matrices)
    /// bias_q, bias_k, bias_v, bias_o: [d_model]
    /// output_idx: 0=output, 1=attn_weights, 2=q_proj, 3=k_proj, 4=v_proj, 5=context
    MultiHeadAttention {
        q_input: ValueId,
        k_input: ValueId,
        v_input: ValueId,
        w_q: ValueId,
        w_k: ValueId,
        w_v: ValueId,
        w_o: ValueId,
        bias_q: ValueId,
        bias_k: ValueId,
        bias_v: ValueId,
        bias_o: ValueId,
        num_heads: usize,
        causal: bool,
        output_idx: usize,
    },
    Dropout {
        input: ValueId,
        ratio: f32,
    },
    Identity(ValueId),
    MaxPoolBackward {
        input: ValueId,
        upstream: ValueId,
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    AvgPoolBackward {
        input: ValueId,
        upstream: ValueId,
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    BatchNormBackwardInput {
        input: ValueId,
        upstream: ValueId,
        weight: ValueId,
        var: ValueId,
    },
    BatchNormBackwardWeight {
        input: ValueId,
        upstream: ValueId,
        mean: ValueId,
        var: ValueId,
    },
    BatchNormBackwardBias {
        upstream: ValueId,
    },
    LayerNormBackwardInput {
        input: ValueId,
        upstream: ValueId,
        weight: ValueId,
        epsilon: f32,
    },
    LayerNormBackwardWeight {
        input: ValueId,
        upstream: ValueId,
        epsilon: f32,
    },
    LayerNormBackwardBias {
        upstream: ValueId,
    },
    Parameter(String),
    Input(String),
    Output(ValueId),
    Phi(Vec<ValueId>),
    Plugin {
        operator: std::sync::Arc<dyn crate::ir::operator::Operator>,
        inputs: Vec<ValueId>,
    },
    /// Low-level backend-specific call.
    /// `target` identifies the backend symbol/kernel name.
    /// `inputs` are the input tensors.
    /// `attrs` are optional string key-value metadata passed to the backend.
    CustomCall {
        target: String,
        inputs: Vec<ValueId>,
        attrs: std::collections::HashMap<String, String>,
    },
    /// Quantize a float tensor to int8 with per-tensor scale/zero_point.
    /// Output is a float tensor holding dequantized values (simulated quantization).
    QuantizeLinear {
        input: ValueId,
        /// Scale factor (f32 scalar)
        scale: f32,
        /// Zero point (i8, stored as i32 for convenience)
        zero_point: i32,
        /// Target bit width: 8 (INT8) or 4 (INT4, future)
        bits: u8,
    },
    /// Dequantize: given a quantized representation (stored as float), just passes through.
    DequantizeLinear {
        input: ValueId,
        scale: f32,
        zero_point: i32,
    },
    /// Depthwise separable convolution (MobileNet-style):
    /// depthwise conv (per-channel) followed by pointwise 1×1 conv.
    /// Input: [N, C, H, W], dw_weight: [C, 1, kH, kW], pw_weight: [C_out, C, 1, 1]
    DepthwiseSeparableConv {
        input: ValueId,
        dw_weight: ValueId,
        pw_weight: ValueId,
        stride: [usize; 2],
        padding: [usize; 2],
    },
    Removed,
}

impl Op {
    #[must_use]
    pub fn input_values(&self) -> Vec<ValueId> {
        match self {
            Op::Add(left, right)
            | Op::Sub(left, right)
            | Op::Mul(left, right)
            | Op::Div(left, right)
            | Op::MatMul(left, right)
            | Op::SigmoidBackward(left, right)
            | Op::GeluBackward(left, right)
            | Op::GeluExactBackward(left, right)
            | Op::ReluBackward(left, right)
            | Op::Conv2D(left, right) => vec![*left, *right],
            Op::Conv2DBackwardInput(input, weight, upstream) => vec![*input, *weight, *upstream],
            Op::Conv2DBackwardWeight(input, weight, upstream) => vec![*input, *weight, *upstream],
            Op::Relu(value) | Op::Softmax(value) | Op::Output(value) => vec![*value],
            Op::Neg(value)
            | Op::Transpose(value)
            | Op::Log(value)
            | Op::Exp(value)
            | Op::Sigmoid(value)
            | Op::Gelu(value)
            | Op::GeluExact(value) => vec![*value],
            Op::GemmBackward { lhs, rhs, bias, .. } => {
                let mut inputs = vec![*lhs, *rhs];
                if let Some(b) = bias {
                    inputs.push(*b);
                }
                inputs
            }
            Op::ElementwiseChain { input, .. } => vec![*input],
            Op::Reshape { input, .. }
            | Op::Gather { input, .. }
            | Op::Slice { input, .. }
            | Op::ReduceSum { input, .. }
            | Op::ReduceMax { input, .. }
            | Op::ReduceMean { input, .. } => {
                vec![*input]
            }
            Op::Gemm { lhs, rhs, bias, .. } => {
                let mut inputs = vec![*lhs, *rhs];
                if let Some(b) = bias {
                    inputs.push(*b);
                }
                inputs
            }
            Op::Concat { inputs, .. } => inputs.clone(),
            Op::Phi(values) => values.clone(),
            Op::Plugin { inputs, .. } => inputs.clone(),
            Op::CustomCall { inputs, .. } => inputs.clone(),
            Op::DepthwiseSeparableConv {
                input,
                dw_weight,
                pw_weight,
                ..
            } => vec![*input, *dw_weight, *pw_weight],
            Op::QuantizeLinear { input, .. } | Op::DequantizeLinear { input, .. } => vec![*input],
            Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::ConstTensor { .. }
            | Op::Parameter(_)
            | Op::Input(_)
            | Op::Removed => Vec::new(),
            Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => vec![*logits, *targets],
            Op::MaxPool { input, .. }
            | Op::AvgPool { input, .. }
            | Op::Flatten { input, .. }
            | Op::GlobalAveragePool { input }
            | Op::Dropout { input, .. }
            | Op::Identity(input) => vec![*input],
            Op::GlobalAveragePoolBackward { input, upstream } => vec![*input, *upstream],
            Op::GroupNorm {
                input,
                weight,
                bias,
                ..
            } => vec![*input, *weight, *bias],
            Op::GroupNormBackwardInput {
                input,
                upstream,
                weight,
                ..
            } => vec![*input, *upstream, *weight],
            Op::GroupNormBackwardWeight {
                input, upstream, ..
            } => vec![*input, *upstream],
            Op::GroupNormBackwardBias { upstream } => vec![*upstream],
            Op::InstanceNorm {
                input,
                weight,
                bias,
                ..
            } => vec![*input, *weight, *bias],
            Op::InstanceNormBackwardInput {
                input,
                upstream,
                weight,
                ..
            } => vec![*input, *upstream, *weight],
            Op::InstanceNormBackwardWeight {
                input, upstream, ..
            } => vec![*input, *upstream],
            Op::InstanceNormBackwardBias { upstream } => vec![*upstream],
            Op::Embedding { weight, indices } => vec![*weight, *indices],
            Op::EmbeddingBackward {
                weight,
                indices,
                upstream,
            } => vec![*weight, *indices, *upstream],
            Op::LstmCell {
                x,
                h_prev,
                c_prev,
                weight_ih,
                weight_hh,
                bias,
                ..
            } => {
                vec![*x, *h_prev, *c_prev, *weight_ih, *weight_hh, *bias]
            }
            Op::LstmCellBackward {
                x,
                h_prev,
                c_prev,
                weight_ih,
                weight_hh,
                gates_raw,
                tanh_c_next,
                dh_next,
                dc_next,
                ..
            } => {
                vec![
                    *x,
                    *h_prev,
                    *c_prev,
                    *weight_ih,
                    *weight_hh,
                    *gates_raw,
                    *tanh_c_next,
                    *dh_next,
                    *dc_next,
                ]
            }
            Op::GruCell {
                x,
                h_prev,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                ..
            } => {
                vec![*x, *h_prev, *weight_ih, *weight_hh, *bias_ih, *bias_hh]
            }
            Op::GruCellBackward {
                x,
                h_prev,
                weight_ih,
                weight_hh,
                z_gate,
                r_gate,
                n_gate,
                dh_next,
                ..
            } => {
                vec![
                    *x, *h_prev, *weight_ih, *weight_hh, *z_gate, *r_gate, *n_gate, *dh_next,
                ]
            }
            Op::ConvTranspose2D { input, weight, .. } => vec![*input, *weight],
            Op::Upsample2D { input, .. } => vec![*input],
            Op::Upsample2DBackward { upstream, .. } => vec![*upstream],
            Op::SinusoidalPE { input } | Op::RoPE { input, .. } => vec![*input],
            Op::RoPEBackward { upstream, .. } => vec![*upstream],
            Op::MultiHeadAttention {
                q_input,
                k_input,
                v_input,
                w_q,
                w_k,
                w_v,
                w_o,
                bias_q,
                bias_k,
                bias_v,
                bias_o,
                ..
            } => {
                vec![
                    *q_input, *k_input, *v_input, *w_q, *w_k, *w_v, *w_o, *bias_q, *bias_k,
                    *bias_v, *bias_o,
                ]
            }
            Op::BatchNorm {
                input,
                weight,
                bias,
                mean,
                var,
            } => vec![*input, *weight, *bias, *mean, *var],
            Op::LayerNorm {
                input,
                weight,
                bias,
                ..
            } => vec![*input, *weight, *bias],
            Op::MaxPoolBackward {
                input, upstream, ..
            }
            | Op::AvgPoolBackward {
                input, upstream, ..
            } => vec![*input, *upstream],
            Op::BatchNormBackwardInput {
                input,
                upstream,
                weight,
                var,
            } => vec![*input, *upstream, *weight, *var],
            Op::ReduceMaxBackward {
                input,
                output_max,
                upstream,
                ..
            } => vec![*input, *output_max, *upstream],
            Op::BatchNormBackwardWeight {
                input,
                upstream,
                mean,
                var,
            } => vec![*input, *upstream, *mean, *var],
            Op::BatchNormBackwardBias { upstream } => vec![*upstream],
            Op::LayerNormBackwardInput {
                input,
                upstream,
                weight,
                ..
            } => vec![*input, *upstream, *weight],
            Op::LayerNormBackwardWeight {
                input, upstream, ..
            } => vec![*input, *upstream],
            Op::LayerNormBackwardBias { upstream } => vec![*upstream],
        }
    }

    pub fn remap_inputs(&mut self, mut remap: impl FnMut(ValueId) -> ValueId) {
        match self {
            Op::Add(left, right)
            | Op::Sub(left, right)
            | Op::Mul(left, right)
            | Op::Div(left, right)
            | Op::MatMul(left, right)
            | Op::ReluBackward(left, right)
            | Op::SigmoidBackward(left, right)
            | Op::GeluBackward(left, right)
            | Op::GeluExactBackward(left, right)
            | Op::Conv2D(left, right) => {
                *left = remap(*left);
                *right = remap(*right);
            }
            Op::Conv2DBackwardInput(input, weight, upstream)
            | Op::Conv2DBackwardWeight(input, weight, upstream) => {
                *input = remap(*input);
                *weight = remap(*weight);
                *upstream = remap(*upstream);
            }
            Op::Relu(value)
            | Op::Softmax(value)
            | Op::Output(value)
            | Op::Neg(value)
            | Op::Transpose(value)
            | Op::Log(value)
            | Op::Exp(value)
            | Op::Sigmoid(value)
            | Op::Gelu(value)
            | Op::GeluExact(value) => {
                *value = remap(*value);
            }
            Op::ElementwiseChain { input, .. } => {
                *input = remap(*input);
            }
            Op::Reshape { input, .. }
            | Op::Gather { input, .. }
            | Op::Slice { input, .. }
            | Op::ReduceSum { input, .. }
            | Op::ReduceMax { input, .. }
            | Op::ReduceMean { input, .. } => {
                *input = remap(*input);
            }
            Op::Gemm { lhs, rhs, bias, .. } => {
                *lhs = remap(*lhs);
                *rhs = remap(*rhs);
                if let Some(b) = bias {
                    *b = remap(*b);
                }
            }
            Op::GemmBackward { lhs, rhs, bias, .. } => {
                *lhs = remap(*lhs);
                *rhs = remap(*rhs);
                if let Some(b) = bias {
                    *b = remap(*b);
                }
            }
            Op::Concat { inputs, .. } => {
                for input in inputs {
                    *input = remap(*input);
                }
            }
            Op::Phi(values) => {
                for value in values {
                    *value = remap(*value);
                }
            }
            Op::Plugin { inputs, .. } => {
                for input in inputs {
                    *input = remap(*input);
                }
            }
            Op::CustomCall { inputs, .. } => {
                for input in inputs {
                    *input = remap(*input);
                }
            }
            Op::DepthwiseSeparableConv {
                input,
                dw_weight,
                pw_weight,
                ..
            } => {
                *input = remap(*input);
                *dw_weight = remap(*dw_weight);
                *pw_weight = remap(*pw_weight);
            }
            Op::QuantizeLinear { input, .. } | Op::DequantizeLinear { input, .. } => {
                *input = remap(*input);
            }
            Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::ConstTensor { .. }
            | Op::Parameter(_)
            | Op::Input(_)
            | Op::Removed => {}
            Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => {
                *logits = remap(*logits);
                *targets = remap(*targets);
            }
            Op::MaxPool { input, .. }
            | Op::AvgPool { input, .. }
            | Op::Flatten { input, .. }
            | Op::GlobalAveragePool { input }
            | Op::Dropout { input, .. }
            | Op::Identity(input) => {
                *input = remap(*input);
            }
            Op::GlobalAveragePoolBackward { input, upstream } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
            }
            Op::GroupNorm {
                input,
                weight,
                bias,
                ..
            } => {
                *input = remap(*input);
                *weight = remap(*weight);
                *bias = remap(*bias);
            }
            Op::GroupNormBackwardInput {
                input,
                upstream,
                weight,
                ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
                *weight = remap(*weight);
            }
            Op::GroupNormBackwardWeight {
                input, upstream, ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
            }
            Op::GroupNormBackwardBias { upstream } => {
                *upstream = remap(*upstream);
            }
            Op::InstanceNorm {
                input,
                weight,
                bias,
                ..
            } => {
                *input = remap(*input);
                *weight = remap(*weight);
                *bias = remap(*bias);
            }
            Op::InstanceNormBackwardInput {
                input,
                upstream,
                weight,
                ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
                *weight = remap(*weight);
            }
            Op::InstanceNormBackwardWeight {
                input, upstream, ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
            }
            Op::InstanceNormBackwardBias { upstream } => {
                *upstream = remap(*upstream);
            }
            Op::Embedding { weight, indices } => {
                *weight = remap(*weight);
                *indices = remap(*indices);
            }
            Op::EmbeddingBackward {
                weight,
                indices,
                upstream,
            } => {
                *weight = remap(*weight);
                *indices = remap(*indices);
                *upstream = remap(*upstream);
            }
            Op::LstmCell {
                x,
                h_prev,
                c_prev,
                weight_ih,
                weight_hh,
                bias,
                ..
            } => {
                *x = remap(*x);
                *h_prev = remap(*h_prev);
                *c_prev = remap(*c_prev);
                *weight_ih = remap(*weight_ih);
                *weight_hh = remap(*weight_hh);
                *bias = remap(*bias);
            }
            Op::LstmCellBackward {
                x,
                h_prev,
                c_prev,
                weight_ih,
                weight_hh,
                gates_raw,
                tanh_c_next,
                dh_next,
                dc_next,
                ..
            } => {
                *x = remap(*x);
                *h_prev = remap(*h_prev);
                *c_prev = remap(*c_prev);
                *weight_ih = remap(*weight_ih);
                *weight_hh = remap(*weight_hh);
                *gates_raw = remap(*gates_raw);
                *tanh_c_next = remap(*tanh_c_next);
                *dh_next = remap(*dh_next);
                *dc_next = remap(*dc_next);
            }
            Op::GruCell {
                x,
                h_prev,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                ..
            } => {
                *x = remap(*x);
                *h_prev = remap(*h_prev);
                *weight_ih = remap(*weight_ih);
                *weight_hh = remap(*weight_hh);
                *bias_ih = remap(*bias_ih);
                *bias_hh = remap(*bias_hh);
            }
            Op::GruCellBackward {
                x,
                h_prev,
                weight_ih,
                weight_hh,
                z_gate,
                r_gate,
                n_gate,
                dh_next,
                ..
            } => {
                *x = remap(*x);
                *h_prev = remap(*h_prev);
                *weight_ih = remap(*weight_ih);
                *weight_hh = remap(*weight_hh);
                *z_gate = remap(*z_gate);
                *r_gate = remap(*r_gate);
                *n_gate = remap(*n_gate);
                *dh_next = remap(*dh_next);
            }
            Op::ConvTranspose2D { input, weight, .. } => {
                *input = remap(*input);
                *weight = remap(*weight);
            }
            Op::Upsample2D { input, .. } => {
                *input = remap(*input);
            }
            Op::Upsample2DBackward { upstream, .. } => {
                *upstream = remap(*upstream);
            }
            Op::SinusoidalPE { input } | Op::RoPE { input, .. } => {
                *input = remap(*input);
            }
            Op::RoPEBackward { upstream, .. } => {
                *upstream = remap(*upstream);
            }
            Op::MultiHeadAttention {
                q_input,
                k_input,
                v_input,
                w_q,
                w_k,
                w_v,
                w_o,
                bias_q,
                bias_k,
                bias_v,
                bias_o,
                ..
            } => {
                *q_input = remap(*q_input);
                *k_input = remap(*k_input);
                *v_input = remap(*v_input);
                *w_q = remap(*w_q);
                *w_k = remap(*w_k);
                *w_v = remap(*w_v);
                *w_o = remap(*w_o);
                *bias_q = remap(*bias_q);
                *bias_k = remap(*bias_k);
                *bias_v = remap(*bias_v);
                *bias_o = remap(*bias_o);
            }
            Op::BatchNorm {
                input,
                weight,
                bias,
                mean,
                var,
            } => {
                *input = remap(*input);
                *weight = remap(*weight);
                *bias = remap(*bias);
                *mean = remap(*mean);
                *var = remap(*var);
            }
            Op::MaxPoolBackward {
                input, upstream, ..
            }
            | Op::AvgPoolBackward {
                input, upstream, ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
            }
            Op::BatchNormBackwardInput {
                input,
                upstream,
                weight,
                var,
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
                *weight = remap(*weight);
                *var = remap(*var);
            }
            Op::ReduceMaxBackward {
                input,
                output_max,
                upstream,
                ..
            } => {
                *input = remap(*input);
                *output_max = remap(*output_max);
                *upstream = remap(*upstream);
            }
            Op::BatchNormBackwardWeight {
                input,
                upstream,
                mean,
                var,
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
                *mean = remap(*mean);
                *var = remap(*var);
            }
            Op::BatchNormBackwardBias { upstream } => {
                *upstream = remap(*upstream);
            }
            Op::LayerNorm {
                input,
                weight,
                bias,
                ..
            } => {
                *input = remap(*input);
                *weight = remap(*weight);
                *bias = remap(*bias);
            }
            Op::LayerNormBackwardInput {
                input,
                upstream,
                weight,
                ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
                *weight = remap(*weight);
            }
            Op::LayerNormBackwardWeight {
                input, upstream, ..
            } => {
                *input = remap(*input);
                *upstream = remap(*upstream);
            }
            Op::LayerNormBackwardBias { upstream } => {
                *upstream = remap(*upstream);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::node::ValueId;

    use super::Op;

    #[test]
    fn gemm_backward_input_values_include_rhs_and_bias() {
        let op = Op::GemmBackward {
            lhs: ValueId(1),
            rhs: ValueId(2),
            bias: Some(ValueId(3)),
            alpha: 1.0,
            beta: 1.0,
        };

        assert_eq!(op.input_values(), vec![ValueId(1), ValueId(2), ValueId(3)]);
    }

    #[test]
    fn gemm_backward_input_values_include_rhs_without_bias() {
        let op = Op::GemmBackward {
            lhs: ValueId(4),
            rhs: ValueId(5),
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        };

        assert_eq!(op.input_values(), vec![ValueId(4), ValueId(5)]);
    }
}

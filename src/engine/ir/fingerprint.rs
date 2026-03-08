use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

use crate::ir::{Graph, Op};

#[must_use]
pub fn graph_fingerprint(graph: &Graph) -> u64 {
    let mut hasher = SipHasher13::new_with_keys(0, 0);
    graph.nodes.len().hash(&mut hasher);
    graph.blocks.len().hash(&mut hasher);
    graph.shape_signature.inputs.hash(&mut hasher);
    graph.shape_signature.parameters.hash(&mut hasher);

    for node in &graph.nodes {
        node.id.0.hash(&mut hasher);
        node.output.0.hash(&mut hasher);
        hash_op(&node.op, &mut hasher);
    }

    hasher.finish()
}

fn hash_op(op: &Op, hasher: &mut SipHasher13) {
    std::mem::discriminant(op).hash(hasher);
    match op {
        Op::ConstInt(value) => value.hash(hasher),
        Op::ConstFloat(value) => value.to_bits().hash(hasher),
        Op::ConstTensor { shape, data } => {
            shape.hash(hasher);
            for value in data {
                value.to_bits().hash(hasher);
            }
        }
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::ReluBackward(a, b)
        | Op::MatMul(a, b)
        | Op::Conv2D(a, b) => {
            a.0.hash(hasher);
            b.0.hash(hasher);
        }
        Op::Conv2DBackwardInput(a, b, c) | Op::Conv2DBackwardWeight(a, b, c) => {
            a.0.hash(hasher);
            b.0.hash(hasher);
            c.0.hash(hasher);
        }
        Op::Neg(v)
        | Op::Transpose(v)
        | Op::Relu(v)
        | Op::Softmax(v)
        | Op::Output(v)
        | Op::Log(v)
        | Op::Exp(v)
        | Op::Sigmoid(v)
        | Op::GeluExact(v)
        | Op::Gelu(v) => {
            v.0.hash(hasher);
        }
        Op::SigmoidBackward(a, b) | Op::GeluBackward(a, b) | Op::GeluExactBackward(a, b) => {
            a.0.hash(hasher);
            b.0.hash(hasher);
        }
        Op::ReduceMaxBackward {
            input,
            output_max,
            upstream,
            axis,
            keepdims,
        } => {
            input.0.hash(hasher);
            output_max.0.hash(hasher);
            upstream.0.hash(hasher);
            axis.hash(hasher);
            keepdims.hash(hasher);
        }
        Op::Gemm {
            lhs,
            rhs,
            bias,
            alpha,
            beta,
        }
        | Op::GemmBackward {
            lhs,
            rhs,
            bias,
            alpha,
            beta,
        } => {
            lhs.0.hash(hasher);
            rhs.0.hash(hasher);
            bias.hash(hasher);
            alpha.to_bits().hash(hasher);
            beta.to_bits().hash(hasher);
        }
        Op::ReduceSum {
            input,
            axis,
            keepdims,
        } => {
            input.0.hash(hasher);
            axis.hash(hasher);
            keepdims.hash(hasher);
        }
        Op::ReduceMax {
            input,
            axis,
            keepdims,
        } => {
            input.0.hash(hasher);
            axis.hash(hasher);
            keepdims.hash(hasher);
        }
        Op::ReduceMean {
            input,
            axis,
            keepdims,
        } => {
            input.0.hash(hasher);
            axis.hash(hasher);
            keepdims.hash(hasher);
        }
        Op::ElementwiseChain { input, ops } => {
            input.0.hash(hasher);
            ops.hash(hasher);
        }
        Op::Reshape { input, shape } => {
            input.0.hash(hasher);
            shape.hash(hasher);
        }
        Op::Concat { inputs, axis } => {
            axis.hash(hasher);
            for input in inputs {
                input.0.hash(hasher);
            }
        }
        Op::Gather {
            input,
            indices,
            axis,
        } => {
            input.0.hash(hasher);
            indices.hash(hasher);
            axis.hash(hasher);
        }
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => {
            input.0.hash(hasher);
            starts.hash(hasher);
            ends.hash(hasher);
            axes.hash(hasher);
        }
        Op::MaxPool {
            input,
            kernel_shape,
            strides,
            pads,
        }
        | Op::AvgPool {
            input,
            kernel_shape,
            strides,
            pads,
        } => {
            input.0.hash(hasher);
            kernel_shape.hash(hasher);
            strides.hash(hasher);
            pads.hash(hasher);
        }
        Op::BatchNorm {
            input,
            weight,
            bias,
            mean,
            var,
        } => {
            input.0.hash(hasher);
            weight.0.hash(hasher);
            bias.0.hash(hasher);
            mean.0.hash(hasher);
            var.0.hash(hasher);
        }
        Op::Flatten { input, axis } => {
            input.0.hash(hasher);
            axis.hash(hasher);
        }
        Op::GlobalAveragePool { input } => {
            input.0.hash(hasher);
        }
        Op::GlobalAveragePoolBackward { input, upstream } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
        }
        Op::GroupNorm {
            input,
            weight,
            bias,
            num_groups,
            epsilon,
        } => {
            input.0.hash(hasher);
            weight.0.hash(hasher);
            bias.0.hash(hasher);
            num_groups.hash(hasher);
            epsilon.to_bits().hash(hasher);
        }
        Op::GroupNormBackwardInput {
            input,
            upstream,
            weight,
            num_groups,
            epsilon,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            weight.0.hash(hasher);
            num_groups.hash(hasher);
            epsilon.to_bits().hash(hasher);
        }
        Op::GroupNormBackwardWeight {
            input,
            upstream,
            num_groups,
            epsilon,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            num_groups.hash(hasher);
            epsilon.to_bits().hash(hasher);
        }
        Op::GroupNormBackwardBias { upstream } => {
            upstream.0.hash(hasher);
        }
        Op::InstanceNorm {
            input,
            weight,
            bias,
            epsilon,
        } => {
            input.0.hash(hasher);
            weight.0.hash(hasher);
            bias.0.hash(hasher);
            epsilon.to_bits().hash(hasher);
        }
        Op::InstanceNormBackwardInput {
            input,
            upstream,
            weight,
            epsilon,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            weight.0.hash(hasher);
            epsilon.to_bits().hash(hasher);
        }
        Op::InstanceNormBackwardWeight {
            input,
            upstream,
            epsilon,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            epsilon.to_bits().hash(hasher);
        }
        Op::InstanceNormBackwardBias { upstream } => {
            upstream.0.hash(hasher);
        }
        Op::Embedding { weight, indices } => {
            weight.0.hash(hasher);
            indices.0.hash(hasher);
        }
        Op::EmbeddingBackward {
            weight,
            indices,
            upstream,
        } => {
            weight.0.hash(hasher);
            indices.0.hash(hasher);
            upstream.0.hash(hasher);
        }
        Op::LstmCell {
            x,
            h_prev,
            c_prev,
            weight_ih,
            weight_hh,
            bias,
            output_idx,
        } => {
            x.0.hash(hasher);
            h_prev.0.hash(hasher);
            c_prev.0.hash(hasher);
            weight_ih.0.hash(hasher);
            weight_hh.0.hash(hasher);
            bias.0.hash(hasher);
            output_idx.hash(hasher);
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
            grad_target,
        } => {
            x.0.hash(hasher);
            h_prev.0.hash(hasher);
            c_prev.0.hash(hasher);
            weight_ih.0.hash(hasher);
            weight_hh.0.hash(hasher);
            gates_raw.0.hash(hasher);
            tanh_c_next.0.hash(hasher);
            dh_next.0.hash(hasher);
            dc_next.0.hash(hasher);
            grad_target.hash(hasher);
        }
        Op::GruCell {
            x,
            h_prev,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            output_idx,
        } => {
            x.0.hash(hasher);
            h_prev.0.hash(hasher);
            weight_ih.0.hash(hasher);
            weight_hh.0.hash(hasher);
            bias_ih.0.hash(hasher);
            bias_hh.0.hash(hasher);
            output_idx.hash(hasher);
        }
        Op::ConvTranspose2D {
            input,
            weight,
            stride,
            padding,
        } => {
            input.0.hash(hasher);
            weight.0.hash(hasher);
            stride.hash(hasher);
            padding.hash(hasher);
        }
        Op::Upsample2D {
            input,
            scale_h,
            scale_w,
            mode,
        } => {
            input.0.hash(hasher);
            scale_h.to_bits().hash(hasher);
            scale_w.to_bits().hash(hasher);
            mode.hash(hasher);
        }
        Op::Upsample2DBackward {
            upstream,
            orig_h,
            orig_w,
            scale_h,
            scale_w,
        } => {
            upstream.0.hash(hasher);
            orig_h.hash(hasher);
            orig_w.hash(hasher);
            scale_h.hash(hasher);
            scale_w.hash(hasher);
        }
        Op::MultiHeadAttention {
            q_input,
            k_input,
            v_input,
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            causal,
            output_idx,
            ..
        } => {
            q_input.0.hash(hasher);
            k_input.0.hash(hasher);
            v_input.0.hash(hasher);
            w_q.0.hash(hasher);
            w_k.0.hash(hasher);
            w_v.0.hash(hasher);
            w_o.0.hash(hasher);
            num_heads.hash(hasher);
            causal.hash(hasher);
            output_idx.hash(hasher);
        }
        Op::SinusoidalPE { input } => {
            input.0.hash(hasher);
        }
        Op::RoPE { input, offset } => {
            input.0.hash(hasher);
            offset.hash(hasher);
        }
        Op::RoPEBackward { upstream, offset } => {
            upstream.0.hash(hasher);
            offset.hash(hasher);
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
            grad_target,
        } => {
            x.0.hash(hasher);
            h_prev.0.hash(hasher);
            weight_ih.0.hash(hasher);
            weight_hh.0.hash(hasher);
            z_gate.0.hash(hasher);
            r_gate.0.hash(hasher);
            n_gate.0.hash(hasher);
            dh_next.0.hash(hasher);
            grad_target.hash(hasher);
        }
        Op::Dropout { input, ratio } => {
            input.0.hash(hasher);
            ratio.to_bits().hash(hasher);
        }
        Op::Identity(v) => {
            v.0.hash(hasher);
        }
        Op::MaxPoolBackward {
            input,
            upstream,
            kernel_shape,
            strides,
            pads,
        }
        | Op::AvgPoolBackward {
            input,
            upstream,
            kernel_shape,
            strides,
            pads,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            kernel_shape.hash(hasher);
            strides.hash(hasher);
            pads.hash(hasher);
        }
        Op::BatchNormBackwardInput {
            input,
            upstream,
            weight,
            var,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            weight.0.hash(hasher);
            var.0.hash(hasher);
        }
        Op::BatchNormBackwardWeight {
            input,
            upstream,
            mean,
            var,
        } => {
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            mean.0.hash(hasher);
            var.0.hash(hasher);
        }
        Op::BatchNormBackwardBias { upstream } => {
            upstream.0.hash(hasher);
        }
        Op::LayerNorm {
            epsilon,
            input,
            weight,
            bias,
        } => {
            epsilon.to_bits().hash(hasher);
            input.0.hash(hasher);
            weight.0.hash(hasher);
            bias.0.hash(hasher);
        }
        Op::LayerNormBackwardInput {
            epsilon,
            input,
            upstream,
            weight,
        } => {
            epsilon.to_bits().hash(hasher);
            input.0.hash(hasher);
            upstream.0.hash(hasher);
            weight.0.hash(hasher);
        }
        Op::LayerNormBackwardWeight {
            epsilon,
            input,
            upstream,
        } => {
            epsilon.to_bits().hash(hasher);
            input.0.hash(hasher);
            upstream.0.hash(hasher);
        }
        Op::LayerNormBackwardBias { upstream } => {
            upstream.0.hash(hasher);
        }
        Op::Parameter(name) | Op::Input(name) => name.hash(hasher),
        Op::Phi(values) => values.iter().for_each(|v| v.0.hash(hasher)),
        Op::Removed => {}
        Op::Plugin { operator, inputs } => {
            operator.fingerprint(hasher);
            inputs.iter().for_each(|v| v.0.hash(hasher));
        }
        Op::CustomCall { target, inputs, .. } => {
            target.hash(hasher);
            inputs.iter().for_each(|v| v.0.hash(hasher));
        }
        Op::DepthwiseSeparableConv {
            input,
            dw_weight,
            pw_weight,
            stride,
            padding,
        } => {
            input.0.hash(hasher);
            dw_weight.0.hash(hasher);
            pw_weight.0.hash(hasher);
            stride.hash(hasher);
            padding.hash(hasher);
        }
        Op::QuantizeLinear {
            input,
            scale,
            zero_point,
            bits,
        } => {
            input.0.hash(hasher);
            scale.to_bits().hash(hasher);
            zero_point.hash(hasher);
            bits.hash(hasher);
        }
        Op::DequantizeLinear {
            input,
            scale,
            zero_point,
        } => {
            input.0.hash(hasher);
            scale.to_bits().hash(hasher);
            zero_point.hash(hasher);
        }
        Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => {
            logits.0.hash(hasher);
            targets.0.hash(hasher);
        }
        Op::MultiHeadAttentionBackward {
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
            attn_weights,
            context,
            upstream,
            num_heads,
            output_idx,
        } => {
            q_input.0.hash(hasher);
            k_input.0.hash(hasher);
            v_input.0.hash(hasher);
            w_q.0.hash(hasher);
            w_k.0.hash(hasher);
            w_v.0.hash(hasher);
            w_o.0.hash(hasher);
            bias_q.0.hash(hasher);
            bias_k.0.hash(hasher);
            bias_v.0.hash(hasher);
            bias_o.0.hash(hasher);
            attn_weights.0.hash(hasher);
            context.0.hash(hasher);
            upstream.0.hash(hasher);
            num_heads.hash(hasher);
            output_idx.hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, graph_fingerprint};

    #[test]
    fn fingerprint_is_deterministic() {
        let mut g1 = Graph::new();
        let b1 = g1.create_block();
        let (_, a1) = g1
            .add_op(b1, Op::ConstInt(1))
            .expect("add op should succeed");
        g1.add_op(b1, Op::Output(a1))
            .expect("add op should succeed");

        let mut g2 = Graph::new();
        let b2 = g2.create_block();
        let (_, a2) = g2
            .add_op(b2, Op::ConstInt(1))
            .expect("add op should succeed");
        g2.add_op(b2, Op::Output(a2))
            .expect("add op should succeed");

        assert_eq!(graph_fingerprint(&g1), graph_fingerprint(&g2));
    }

    #[test]
    fn fingerprint_different_graphs_differ() {
        let mut g1 = Graph::new();
        let b1 = g1.create_block();
        let (_, a1) = g1
            .add_op(b1, Op::ConstInt(1))
            .expect("add op should succeed");
        g1.add_op(b1, Op::Output(a1))
            .expect("add op should succeed");

        let mut g2 = Graph::new();
        let b2 = g2.create_block();
        let (_, a2) = g2
            .add_op(b2, Op::ConstInt(2))
            .expect("add op should succeed");
        g2.add_op(b2, Op::Output(a2))
            .expect("add op should succeed");

        assert_ne!(graph_fingerprint(&g1), graph_fingerprint(&g2));
    }
}

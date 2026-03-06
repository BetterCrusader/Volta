/// Transformer building blocks for Volta IR graphs.
///
/// These functions compose existing Op primitives to build standard transformer
/// components. They work by adding nodes to a Graph + BasicBlock.
use crate::ir::{Graph, GraphError, op::Op, node::ValueId, block::BasicBlockId};

/// Configuration for a TransformerEncoder block.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    /// Feed-forward inner dimension (typically 4 * d_model)
    pub ffn_dim: usize,
    pub dropout: f32,
    pub causal: bool,
    pub epsilon: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            ffn_dim: 2048,
            dropout: 0.0,
            causal: false,
            epsilon: 1e-5,
        }
    }
}

/// Adds a TransformerEncoder block to the graph.
///
/// Architecture (Post-LN):
/// ```text
///   x -> MHA(x, x, x) -> +x -> LayerNorm -> FFN -> +x -> LayerNorm
/// ```
///
/// Parameters:
/// - `x`: input node [batch, seq, d_model]
/// - `w_q, w_k, w_v, w_o`: MHA weight matrices [d_model, d_model]
/// - `bq, bk, bv, bo`: MHA bias vectors [d_model]
/// - `ln1_w, ln1_b`: first LayerNorm weight/bias [d_model]
/// - `ffn_w1, ffn_b1`: FFN first linear [ffn_dim, d_model] / [ffn_dim]
/// - `ffn_w2, ffn_b2`: FFN second linear [d_model, ffn_dim] / [d_model]
/// - `ln2_w, ln2_b`: second LayerNorm weight/bias [d_model]
///
/// Returns: output ValueId [batch, seq, d_model]
#[allow(clippy::too_many_arguments)]
pub fn add_transformer_encoder_block(
    graph: &mut Graph,
    block: BasicBlockId,
    x: ValueId,
    // MHA params
    w_q: ValueId, w_k: ValueId, w_v: ValueId, w_o: ValueId,
    bq: ValueId, bk: ValueId, bv: ValueId, bo: ValueId,
    // First LayerNorm params
    ln1_w: ValueId, ln1_b: ValueId,
    // FFN params
    ffn_w1: ValueId, ffn_b1: ValueId,
    ffn_w2: ValueId, ffn_b2: ValueId,
    // Second LayerNorm params
    ln2_w: ValueId, ln2_b: ValueId,
    config: &TransformerConfig,
) -> Result<ValueId, GraphError> {
    // 1. Self-attention (output_idx=0 = final MHA output)
    let (_, attn_out) = graph.add_op(block, Op::MultiHeadAttention {
        q_input: x, k_input: x, v_input: x,
        w_q, w_k, w_v, w_o,
        bias_q: bq, bias_k: bk, bias_v: bv, bias_o: bo,
        num_heads: config.num_heads,
        causal: config.causal,
        output_idx: 0,
    })?;

    // 2. Residual + LayerNorm 1
    // Note: For LN to work on 3D tensors [batch, seq, d_model], we need to reshape.
    // Our LayerNorm kernel expects [N, D]. We use Reshape to flatten batch*seq.
    // (Reshape back at the end)
    let (_, residual1) = graph.add_op(block, Op::Add(x, attn_out))?;

    // Reshape to [batch*seq, d_model] for LayerNorm
    let (_, ln1_in) = graph.add_op(block, Op::Reshape {
        input: residual1,
        shape: vec![0, config.d_model], // 0 = dynamic (batch*seq)
    })?;
    let (_, ln1_out) = graph.add_op(block, Op::LayerNorm {
        input: ln1_in,
        weight: ln1_w,
        bias: ln1_b,
        epsilon: config.epsilon,
    })?;
    // Reshape back (use Identity to keep the value; actual shape tracked by shape inference)
    let (_, ln1_out_3d) = graph.add_op(block, Op::Identity(ln1_out))?;

    // 3. FFN: Linear → GELU → Linear (using Gemm for the linear projections)
    // FFN W1: [batch*seq, d_model] @ [d_model, ffn_dim] = [batch*seq, ffn_dim]
    let (_, ffn1) = graph.add_op(block, Op::Gemm {
        lhs: ln1_out_3d,
        rhs: ffn_w1,
        bias: Some(ffn_b1),
        alpha: 1.0,
        beta: 1.0,
    })?;
    let (_, ffn1_act) = graph.add_op(block, Op::Gelu(ffn1))?;

    // Optional dropout
    let ffn1_drop = if config.dropout > 0.0 {
        let (_, d) = graph.add_op(block, Op::Dropout { input: ffn1_act, ratio: config.dropout })?;
        d
    } else {
        ffn1_act
    };

    // FFN W2: [batch*seq, ffn_dim] @ [ffn_dim, d_model] = [batch*seq, d_model]
    let (_, ffn2) = graph.add_op(block, Op::Gemm {
        lhs: ffn1_drop,
        rhs: ffn_w2,
        bias: Some(ffn_b2),
        alpha: 1.0,
        beta: 1.0,
    })?;

    // 4. Residual + LayerNorm 2
    let (_, residual2) = graph.add_op(block, Op::Add(ln1_out_3d, ffn2))?;
    let (_, ln2_out) = graph.add_op(block, Op::LayerNorm {
        input: residual2,
        weight: ln2_w,
        bias: ln2_b,
        epsilon: config.epsilon,
    })?;

    Ok(ln2_out)
}

/// Adds a causal decoder-style transformer block.
/// Same as encoder but with causal=true in MHA.
#[allow(clippy::too_many_arguments)]
pub fn add_transformer_decoder_block(
    graph: &mut Graph,
    block: BasicBlockId,
    x: ValueId,
    w_q: ValueId, w_k: ValueId, w_v: ValueId, w_o: ValueId,
    bq: ValueId, bk: ValueId, bv: ValueId, bo: ValueId,
    ln1_w: ValueId, ln1_b: ValueId,
    ffn_w1: ValueId, ffn_b1: ValueId,
    ffn_w2: ValueId, ffn_b2: ValueId,
    ln2_w: ValueId, ln2_b: ValueId,
    config: &TransformerConfig,
) -> Result<ValueId, GraphError> {
    let causal_config = TransformerConfig { causal: true, ..config.clone() };
    add_transformer_encoder_block(
        graph, block, x,
        w_q, w_k, w_v, w_o,
        bq, bk, bv, bo,
        ln1_w, ln1_b,
        ffn_w1, ffn_b1, ffn_w2, ffn_b2,
        ln2_w, ln2_b,
        &causal_config,
    )
}

/// Configuration for Vision Transformer (ViT).
#[derive(Debug, Clone)]
pub struct VitConfig {
    /// Image size (assumed square).
    pub image_size: usize,
    /// Patch size (assumed square).
    pub patch_size: usize,
    /// Number of input channels (3 for RGB).
    pub in_channels: usize,
    /// Embedding dimension (d_model).
    pub d_model: usize,
    /// Number of transformer encoder layers.
    pub depth: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feed-forward inner dimension (typically 4 * d_model).
    pub mlp_dim: usize,
    /// Number of output classes (for classification head).
    pub num_classes: usize,
    /// Dropout probability.
    pub dropout: f32,
}

impl Default for VitConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            d_model: 768,
            depth: 12,
            num_heads: 12,
            mlp_dim: 3072,
            num_classes: 1000,
            dropout: 0.0,
        }
    }
}

/// Weights needed for one ViT encoder layer.
pub struct VitLayerWeights {
    pub w_q: ValueId, pub w_k: ValueId, pub w_v: ValueId, pub w_o: ValueId,
    pub bq: ValueId, pub bk: ValueId, pub bv: ValueId, pub bo: ValueId,
    pub ln1_w: ValueId, pub ln1_b: ValueId,
    pub ffn_w1: ValueId, pub ffn_b1: ValueId,
    pub ffn_w2: ValueId, pub ffn_b2: ValueId,
    pub ln2_w: ValueId, pub ln2_b: ValueId,
}

/// All weights needed to build a ViT graph.
pub struct VitWeights {
    /// Patch embedding linear layer: [d_model, patch_size*patch_size*in_channels]
    pub patch_embed_w: ValueId,
    /// Patch embedding bias: [d_model]
    pub patch_embed_b: ValueId,
    /// CLS token: [1, 1, d_model]
    pub cls_token: ValueId,
    /// Position embedding: [1, num_patches+1, d_model]
    pub pos_embed: ValueId,
    /// LayerNorm before classification head
    pub norm_w: ValueId,
    pub norm_b: ValueId,
    /// Classification head linear: [num_classes, d_model]
    pub head_w: ValueId,
    /// Classification head bias: [num_classes]
    pub head_b: ValueId,
    /// One entry per transformer layer
    pub layers: Vec<VitLayerWeights>,
}

/// Build a Vision Transformer (ViT) image classification model.
///
/// Pipeline:
///   patches → patch_embed → + cls_token → + pos_embed → N × Transformer → norm → head → logits
///
/// Parameters:
/// - `x`: input patches [batch, num_patches, patch_dim] (pre-patchified for simplicity)
///
/// Returns: logit ValueId [batch, num_classes]
pub fn add_vit(
    graph: &mut Graph,
    block: BasicBlockId,
    x: ValueId,
    weights: &VitWeights,
    config: &VitConfig,
) -> Result<ValueId, GraphError> {
    let transformer_config = TransformerConfig {
        d_model: config.d_model,
        num_heads: config.num_heads,
        ffn_dim: config.mlp_dim,
        dropout: config.dropout,
        causal: false,
        epsilon: 1e-6,
    };

    // Patch embedding: Gemm(x, patch_embed_w^T) + patch_embed_b
    let (_, patch_emb) = graph.add_op(block, Op::Gemm {
        lhs: x,
        rhs: weights.patch_embed_w,
        bias: Some(weights.patch_embed_b),
        alpha: 1.0,
        beta: 1.0,
    })?;

    // Prepend CLS token: Concat([cls_token, patch_emb], axis=1)
    let (_, tokens) = graph.add_op(block, Op::Concat {
        inputs: vec![weights.cls_token, patch_emb],
        axis: 1,
    })?;

    // Add positional embedding
    let (_, tokens) = graph.add_op(block, Op::Add(tokens, weights.pos_embed))?;

    // N transformer encoder layers
    let mut hidden = tokens;
    for layer in &weights.layers {
        hidden = add_transformer_encoder_block(
            graph, block, hidden,
            layer.w_q, layer.w_k, layer.w_v, layer.w_o,
            layer.bq, layer.bk, layer.bv, layer.bo,
            layer.ln1_w, layer.ln1_b,
            layer.ffn_w1, layer.ffn_b1, layer.ffn_w2, layer.ffn_b2,
            layer.ln2_w, layer.ln2_b,
            &transformer_config,
        )?;
    }

    // Layer norm
    let (_, normed) = graph.add_op(block, Op::LayerNorm {
        input: hidden,
        weight: weights.norm_w,
        bias: weights.norm_b,
        epsilon: transformer_config.epsilon,
    })?;

    // Extract CLS token (index 0 along sequence dim)
    let (_, cls_out) = graph.add_op(block, Op::Gather {
        input: normed,
        indices: vec![0],
        axis: 1,
    })?;

    // Classification head
    let (_, logits) = graph.add_op(block, Op::Gemm {
        lhs: cls_out,
        rhs: weights.head_w,
        bias: Some(weights.head_b),
        alpha: 1.0,
        beta: 1.0,
    })?;

    Ok(logits)
}

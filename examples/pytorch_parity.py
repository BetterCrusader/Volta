import json
import math
import sys

import torch
import torch.nn.functional as F


def mlp_case():
    x = torch.tensor(
        [[0.2, -0.1, 0.3], [0.7, 0.5, -0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w1 = torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.5, 0.6, -0.7, 0.8],
            [-0.9, 1.0, 0.2, -0.3],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    b1 = torch.tensor([0.05, -0.1, 0.15, 0.2], dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(
        [[0.2, -0.4], [0.1, 0.3], [-0.5, 0.7], [0.6, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    b2 = torch.tensor([0.25, -0.35], dtype=torch.float32, requires_grad=True)

    hidden = torch.relu(torch.matmul(x, w1) + b1)
    out = torch.matmul(hidden, w2) + b2
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "x": x.grad.reshape(-1).tolist(),
            "w1": w1.grad.reshape(-1).tolist(),
            "b1": b1.grad.reshape(-1).tolist(),
            "w2": w2.grad.reshape(-1).tolist(),
            "b2": b2.grad.reshape(-1).tolist(),
        },
    }


def conv_case():
    x = torch.tensor(
        [[0.2, -0.1, 0.3], [0.7, 0.5, -0.4], [0.6, -0.2, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w = torch.tensor([[0.5, -0.25], [0.75, 0.1]], dtype=torch.float32, requires_grad=True)

    out = F.conv2d(
        x.view(1, 1, 3, 3),
        w.view(1, 1, 2, 2),
    ).view(2, 2)
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "x": x.grad.reshape(-1).tolist(),
            "w": w.grad.reshape(-1).tolist(),
        },
    }


def layernorm_case():
    x = torch.tensor(
        [[0.2, -0.1, 0.3, 0.4, -0.2], [0.7, 0.5, -0.4, 0.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w = torch.tensor([1.1, 0.9, -0.7, 0.5, 0.3], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([0.05, -0.15, 0.2, -0.1, 0.25], dtype=torch.float32, requires_grad=True)

    out = F.layer_norm(x, (5,), w, b, 1e-5)
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "x": x.grad.reshape(-1).tolist(),
            "w": w.grad.reshape(-1).tolist(),
            "b": b.grad.reshape(-1).tolist(),
        },
    }


def batchnorm_case():
    x = torch.tensor(
        [[[[0.2, -0.1], [0.3, 0.4]], [[0.7, 0.5], [-0.4, 0.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w = torch.tensor([1.2, -0.8], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([0.1, -0.2], dtype=torch.float32, requires_grad=True)
    mean = torch.tensor([0.15, 0.05], dtype=torch.float32)
    var = torch.tensor([0.25, 0.5], dtype=torch.float32)

    out = F.batch_norm(x, mean, var, w, b, training=False, momentum=0.1, eps=1e-5)
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "x": x.grad.reshape(-1).tolist(),
            "w": w.grad.reshape(-1).tolist(),
            "b": b.grad.reshape(-1).tolist(),
        },
    }


def transformer_case():
    x = torch.tensor(
        [[[0.2, -0.1, 0.3, 0.4], [0.7, 0.5, -0.4, 0.1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_q = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.4],
            [-0.5, 0.6, 0.1, -0.2],
            [0.7, 0.2, -0.3, 0.5],
            [0.4, -0.6, 0.8, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_k = torch.tensor(
        [
            [0.1, 0.2, -0.4, 0.3],
            [0.5, -0.7, 0.6, 0.2],
            [-0.3, 0.8, 0.4, -0.1],
            [0.2, 0.1, 0.5, -0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_v = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.7],
            [0.6, 0.4, -0.5, 0.2],
            [0.2, -0.8, 0.9, 0.1],
            [-0.4, 0.3, 0.2, 0.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_o = torch.tensor(
        [
            [0.4, -0.3, 0.2, 0.1],
            [0.5, 0.6, -0.7, 0.2],
            [-0.1, 0.8, 0.3, -0.4],
            [0.7, -0.2, 0.5, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    b_q = torch.tensor([0.05, -0.1, 0.15, -0.2], dtype=torch.float32, requires_grad=True)
    b_k = torch.tensor([-0.05, 0.2, -0.15, 0.1], dtype=torch.float32, requires_grad=True)
    b_v = torch.tensor([0.1, 0.05, -0.2, 0.25], dtype=torch.float32, requires_grad=True)
    b_o = torch.tensor([-0.1, 0.15, 0.05, -0.05], dtype=torch.float32, requires_grad=True)
    ln1_w = torch.tensor([1.0, 0.9, 1.1, -0.8], dtype=torch.float32, requires_grad=True)
    ln1_b = torch.tensor([0.05, -0.1, 0.15, 0.2], dtype=torch.float32, requires_grad=True)
    ffn_w1 = torch.tensor(
        [
            [0.2, -0.3, 0.1, 0.5, 0.4, -0.2],
            [0.6, 0.7, -0.5, 0.2, -0.1, 0.3],
            [-0.4, 0.8, 0.9, -0.6, 0.2, 0.1],
            [0.3, -0.7, 0.4, 0.5, -0.8, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    ffn_b1 = torch.tensor([0.1, -0.2, 0.05, 0.15, -0.1, 0.2], dtype=torch.float32, requires_grad=True)
    ffn_w2 = torch.tensor(
        [
            [0.3, -0.4, 0.2, 0.1],
            [0.5, 0.6, -0.7, 0.2],
            [-0.1, 0.8, 0.3, -0.4],
            [0.7, -0.2, 0.5, 0.6],
            [0.4, 0.1, -0.3, 0.2],
            [-0.5, 0.9, 0.6, -0.7],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    ffn_b2 = torch.tensor([0.2, -0.15, 0.05, 0.1], dtype=torch.float32, requires_grad=True)
    ln2_w = torch.tensor([0.95, -1.05, 0.85, 1.1], dtype=torch.float32, requires_grad=True)
    ln2_b = torch.tensor([-0.05, 0.1, -0.15, 0.2], dtype=torch.float32, requires_grad=True)

    batch = 1
    seq = 2
    d_model = 4
    num_heads = 2
    head_dim = d_model // num_heads

    q_proj = torch.matmul(x, w_q.transpose(0, 1)) + b_q
    k_proj = torch.matmul(x, w_k.transpose(0, 1)) + b_k
    v_proj = torch.matmul(x, w_v.transpose(0, 1)) + b_v

    q_heads = q_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k_heads = k_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    v_heads = v_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, v_heads)
    context = context.transpose(1, 2).contiguous().view(batch, seq, d_model)
    attn_out = torch.matmul(context, w_o.transpose(0, 1)) + b_o

    residual1 = x + attn_out
    ln1_in = residual1.view(batch * seq, d_model)
    ln1_out = F.layer_norm(ln1_in, (d_model,), ln1_w, ln1_b, 1e-5)
    ffn1 = torch.matmul(ln1_out, ffn_w1) + ffn_b1
    ffn1_act = F.gelu(ffn1, approximate="tanh")
    ffn2 = torch.matmul(ffn1_act, ffn_w2) + ffn_b2
    residual2 = ln1_out + ffn2
    out = F.layer_norm(residual2, (d_model,), ln2_w, ln2_b, 1e-5)
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "x": x.grad.reshape(-1).tolist(),
            "w_q": w_q.grad.reshape(-1).tolist(),
            "ln1_w": ln1_w.grad.reshape(-1).tolist(),
            "ffn_w1": ffn_w1.grad.reshape(-1).tolist(),
            "ln2_w": ln2_w.grad.reshape(-1).tolist(),
        },
    }


def mlp_train_sgd_case():
    x = torch.tensor(
        [[0.2, -0.1, 0.3], [0.7, 0.5, -0.4]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[0.1, -0.2], [0.3, 0.4]],
        dtype=torch.float32,
    )
    w1 = torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.5, 0.6, -0.7, 0.8],
            [-0.9, 1.0, 0.2, -0.3],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    b1 = torch.tensor([0.05, -0.1, 0.15, 0.2], dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(
        [[0.2, -0.4], [0.1, 0.3], [-0.5, 0.7], [0.6, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    b2 = torch.tensor([0.25, -0.35], dtype=torch.float32, requires_grad=True)
    lr = 0.05

    hidden = torch.relu(torch.matmul(x, w1) + b1)
    out = torch.matmul(hidden, w2) + b2
    diff = out - target
    loss = torch.mean(diff * diff)
    loss.backward()

    grads = {
        "w1": w1.grad.detach().reshape(-1).tolist(),
        "b1": b1.grad.detach().reshape(-1).tolist(),
        "w2": w2.grad.detach().reshape(-1).tolist(),
        "b2": b2.grad.detach().reshape(-1).tolist(),
    }

    with torch.no_grad():
        w1 -= lr * w1.grad
        b1 -= lr * b1.grad
        w2 -= lr * w2.grad
        b2 -= lr * b2.grad

    hidden_after = torch.relu(torch.matmul(x, w1) + b1)
    out_after = torch.matmul(hidden_after, w2) + b2
    diff_after = out_after - target
    loss_after = torch.mean(diff_after * diff_after)

    return {
        "loss_before": float(loss.detach().item()),
        "loss_after": float(loss_after.detach().item()),
        "gradients": grads,
        "updated_parameters": {
            "w1": w1.detach().reshape(-1).tolist(),
            "b1": b1.detach().reshape(-1).tolist(),
            "w2": w2.detach().reshape(-1).tolist(),
            "b2": b2.detach().reshape(-1).tolist(),
        },
    }


def mlp_train_loop_sgd_case():
    dataset = [
        (
            torch.tensor([[0.2, -0.1, 0.3], [0.7, 0.5, -0.4]], dtype=torch.float32),
            torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32),
        ),
        (
            torch.tensor([[0.4, 0.2, -0.5], [-0.3, 0.6, 0.8]], dtype=torch.float32),
            torch.tensor([[0.2, 0.05], [-0.1, 0.6]], dtype=torch.float32),
        ),
    ]
    w1 = torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.5, 0.6, -0.7, 0.8],
            [-0.9, 1.0, 0.2, -0.3],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    b1 = torch.tensor([0.05, -0.1, 0.15, 0.2], dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(
        [[0.2, -0.4], [0.1, 0.3], [-0.5, 0.7], [0.6, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    b2 = torch.tensor([0.25, -0.35], dtype=torch.float32, requires_grad=True)
    lr = 0.05
    epochs = 3

    for _ in range(epochs):
        for x, target in dataset:
            if w1.grad is not None:
                w1.grad.zero_()
                b1.grad.zero_()
                w2.grad.zero_()
                b2.grad.zero_()
            hidden = torch.relu(torch.matmul(x, w1) + b1)
            out = torch.matmul(hidden, w2) + b2
            diff = out - target
            loss = torch.mean(diff * diff)
            loss.backward()
            with torch.no_grad():
                w1 -= lr * w1.grad
                b1 -= lr * b1.grad
                w2 -= lr * w2.grad
                b2 -= lr * b2.grad

    last_x, last_target = dataset[-1]
    hidden = torch.relu(torch.matmul(last_x, w1) + b1)
    out = torch.matmul(hidden, w2) + b2
    diff = out - last_target
    final_loss = torch.mean(diff * diff)

    return {
        "final_loss": float(final_loss.detach().item()),
        "final_parameters": {
            "w1": w1.detach().reshape(-1).tolist(),
            "b1": b1.detach().reshape(-1).tolist(),
            "w2": w2.detach().reshape(-1).tolist(),
            "b2": b2.detach().reshape(-1).tolist(),
        },
    }


def mha_self_case():
    batch = 1
    seq = 2
    d_model = 4
    num_heads = 2
    head_dim = d_model // num_heads

    x = torch.tensor(
        [[[0.2, -0.1, 0.3, 0.4], [0.7, 0.5, -0.4, 0.1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_q = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.4],
            [-0.5, 0.6, 0.1, -0.2],
            [0.7, 0.2, -0.3, 0.5],
            [0.4, -0.6, 0.8, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_k = torch.tensor(
        [
            [0.1, 0.2, -0.4, 0.3],
            [0.5, -0.7, 0.6, 0.2],
            [-0.3, 0.8, 0.4, -0.1],
            [0.2, 0.1, 0.5, -0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_v = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.7],
            [0.6, 0.4, -0.5, 0.2],
            [0.2, -0.8, 0.9, 0.1],
            [-0.4, 0.3, 0.2, 0.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_o = torch.tensor(
        [
            [0.4, -0.3, 0.2, 0.1],
            [0.5, 0.6, -0.7, 0.2],
            [-0.1, 0.8, 0.3, -0.4],
            [0.7, -0.2, 0.5, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    b_q = torch.tensor([0.05, -0.1, 0.15, -0.2], dtype=torch.float32, requires_grad=True)
    b_k = torch.tensor([-0.05, 0.2, -0.15, 0.1], dtype=torch.float32, requires_grad=True)
    b_v = torch.tensor([0.1, 0.05, -0.2, 0.25], dtype=torch.float32, requires_grad=True)
    b_o = torch.tensor([-0.1, 0.15, 0.05, -0.05], dtype=torch.float32, requires_grad=True)

    q_proj = torch.matmul(x, w_q.transpose(0, 1)) + b_q
    k_proj = torch.matmul(x, w_k.transpose(0, 1)) + b_k
    v_proj = torch.matmul(x, w_v.transpose(0, 1)) + b_v

    q_heads = q_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k_heads = k_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    v_heads = v_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, v_heads)
    context = context.transpose(1, 2).contiguous().view(batch, seq, d_model)
    out = torch.matmul(context, w_o.transpose(0, 1)) + b_o
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "x": x.grad.reshape(-1).tolist(),
            "w_q": w_q.grad.reshape(-1).tolist(),
            "w_o": w_o.grad.reshape(-1).tolist(),
        },
    }


def mha_case():
    batch = 1
    seq = 2
    d_model = 4
    num_heads = 2
    head_dim = d_model // num_heads

    q = torch.tensor(
        [[[0.2, -0.1, 0.3, 0.4], [0.7, 0.5, -0.4, 0.1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    k = torch.tensor(
        [[[0.6, -0.2, 0.5, -0.3], [0.1, 0.8, -0.7, 0.2]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    v = torch.tensor(
        [[[0.3, 0.4, -0.5, 0.2], [0.9, -0.6, 0.1, 0.7]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_q = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.4],
            [-0.5, 0.6, 0.1, -0.2],
            [0.7, 0.2, -0.3, 0.5],
            [0.4, -0.6, 0.8, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_k = torch.tensor(
        [
            [0.1, 0.2, -0.4, 0.3],
            [0.5, -0.7, 0.6, 0.2],
            [-0.3, 0.8, 0.4, -0.1],
            [0.2, 0.1, 0.5, -0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_v = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.7],
            [0.6, 0.4, -0.5, 0.2],
            [0.2, -0.8, 0.9, 0.1],
            [-0.4, 0.3, 0.2, 0.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_o = torch.tensor(
        [
            [0.4, -0.3, 0.2, 0.1],
            [0.5, 0.6, -0.7, 0.2],
            [-0.1, 0.8, 0.3, -0.4],
            [0.7, -0.2, 0.5, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    b_q = torch.tensor([0.05, -0.1, 0.15, -0.2], dtype=torch.float32, requires_grad=True)
    b_k = torch.tensor([-0.05, 0.2, -0.15, 0.1], dtype=torch.float32, requires_grad=True)
    b_v = torch.tensor([0.1, 0.05, -0.2, 0.25], dtype=torch.float32, requires_grad=True)
    b_o = torch.tensor([-0.1, 0.15, 0.05, -0.05], dtype=torch.float32, requires_grad=True)

    q_proj = torch.matmul(q, w_q.transpose(0, 1)) + b_q
    k_proj = torch.matmul(k, w_k.transpose(0, 1)) + b_k
    v_proj = torch.matmul(v, w_v.transpose(0, 1)) + b_v

    q_heads = q_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k_heads = k_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    v_heads = v_proj.view(batch, seq, num_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, v_heads)
    context = context.transpose(1, 2).contiguous().view(batch, seq, d_model)
    out = torch.matmul(context, w_o.transpose(0, 1)) + b_o
    loss = out.sum()
    loss.backward()

    return {
        "output_shape": list(out.shape),
        "output": out.detach().reshape(-1).tolist(),
        "gradients": {
            "q": q.grad.reshape(-1).tolist(),
            "k": k.grad.reshape(-1).tolist(),
            "v": v.grad.reshape(-1).tolist(),
            "w_q": w_q.grad.reshape(-1).tolist(),
            "w_k": w_k.grad.reshape(-1).tolist(),
            "w_o": w_o.grad.reshape(-1).tolist(),
        },
    }


def main():
    torch.set_num_threads(1)
    case = sys.argv[1]
    if case == "mlp":
        result = mlp_case()
    elif case == "conv2d":
        result = conv_case()
    elif case == "layernorm":
        result = layernorm_case()
    elif case == "batchnorm":
        result = batchnorm_case()
    elif case == "transformer":
        result = transformer_case()
    elif case == "mlp_train_sgd":
        result = mlp_train_sgd_case()
    elif case == "mlp_train_loop_sgd":
        result = mlp_train_loop_sgd_case()
    elif case == "mha_self":
        result = mha_self_case()
    elif case == "mha":
        result = mha_case()
    else:
        raise SystemExit(f"unknown case: {case}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

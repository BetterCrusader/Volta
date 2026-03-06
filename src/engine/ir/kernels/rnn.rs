use crate::engine::ir::tensor::{Tensor, TensorError};

// ─── Helper: sigmoid ──────────────────────────────────────────────────────────

#[inline]
fn sigmoid_f32(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ─── LSTM cell ────────────────────────────────────────────────────────────────
// Inputs:
//   x:          [batch, input_size]
//   h_prev:     [batch, hidden_size]
//   c_prev:     [batch, hidden_size]
//   weight_ih:  [4*hidden, input_size]   — maps input to gates
//   weight_hh:  [4*hidden, hidden_size]  — maps h_prev to gates
//   bias:       [4*hidden]               — gate bias (optional, pass zeros if absent)
// Outputs (as separate tensors in a struct):
//   h_next, c_next, gates (for backward)

pub struct LstmCellOutput {
    pub h_next: Tensor,
    pub c_next: Tensor,
    /// Pre-activation gate values [batch, 4*hidden]: [i_raw, f_raw, g_raw, o_raw]
    pub gates_raw: Tensor,
    /// Activated cell-candidate tanh(c_next) [batch, hidden]
    pub tanh_c_next: Tensor,
}

pub fn lstm_cell(
    x: &Tensor,
    h_prev: &Tensor,
    c_prev: &Tensor,
    weight_ih: &Tensor,
    weight_hh: &Tensor,
    bias: &Tensor,
) -> Result<LstmCellOutput, TensorError> {
    let x_c = x.make_contiguous()?;
    let h_c = h_prev.make_contiguous()?;
    let cp = c_prev.make_contiguous()?;
    let wih = weight_ih.make_contiguous()?;
    let whh = weight_hh.make_contiguous()?;
    let b = bias.make_contiguous()?;

    if x_c.shape.len() != 2 || h_c.shape.len() != 2 {
        return Err(TensorError { message: "LSTM cell expects rank-2 x and h_prev".to_string() });
    }
    let batch = x_c.shape[0];
    let input_size = x_c.shape[1];
    let hidden = h_c.shape[1];
    if cp.shape != h_c.shape {
        return Err(TensorError { message: "LSTM: c_prev shape must match h_prev".to_string() });
    }
    if wih.shape != vec![4 * hidden, input_size] {
        return Err(TensorError { message: format!(
            "LSTM weight_ih must be [{}, {}], got {:?}", 4*hidden, input_size, wih.shape
        )});
    }
    if whh.shape != vec![4 * hidden, hidden] {
        return Err(TensorError { message: format!(
            "LSTM weight_hh must be [{}, {}], got {:?}", 4*hidden, hidden, whh.shape
        )});
    }

    let mut gates_raw = vec![0.0_f32; batch * 4 * hidden];
    let mut h_next = vec![0.0_f32; batch * hidden];
    let mut c_next = vec![0.0_f32; batch * hidden];
    let mut tanh_c = vec![0.0_f32; batch * hidden];

    for n in 0..batch {
        // Compute gates = W_ih @ x[n] + W_hh @ h_prev[n] + bias
        for g in 0..4 * hidden {
            let mut v = b.data[g] as f64;
            // W_ih contribution
            for k in 0..input_size {
                v += wih.data[g * input_size + k] as f64 * x_c.data[n * input_size + k] as f64;
            }
            // W_hh contribution
            for k in 0..hidden {
                v += whh.data[g * hidden + k] as f64 * h_c.data[n * hidden + k] as f64;
            }
            gates_raw[n * 4 * hidden + g] = v as f32;
        }

        // Activate gates: i=sigmoid, f=sigmoid, g=tanh, o=sigmoid
        for j in 0..hidden {
            let i_gate = sigmoid_f32(gates_raw[n * 4 * hidden + j]);
            let f_gate = sigmoid_f32(gates_raw[n * 4 * hidden + hidden + j]);
            let g_gate = gates_raw[n * 4 * hidden + 2 * hidden + j].tanh();
            let o_gate = sigmoid_f32(gates_raw[n * 4 * hidden + 3 * hidden + j]);

            let c_new = f_gate * cp.data[n * hidden + j] + i_gate * g_gate;
            let tc = c_new.tanh();
            c_next[n * hidden + j] = c_new;
            tanh_c[n * hidden + j] = tc;
            h_next[n * hidden + j] = o_gate * tc;
        }
    }

    Ok(LstmCellOutput {
        h_next: Tensor::new(vec![batch, hidden], h_next)?,
        c_next: Tensor::new(vec![batch, hidden], c_next)?,
        gates_raw: Tensor::new(vec![batch, 4 * hidden], gates_raw)?,
        tanh_c_next: Tensor::new(vec![batch, hidden], tanh_c)?,
    })
}

/// LSTM cell backward.
/// Returns gradients for: x, h_prev, c_prev, weight_ih, weight_hh, bias.
pub struct LstmCellGrads {
    pub dx: Tensor,
    pub dh_prev: Tensor,
    pub dc_prev: Tensor,
    pub dweight_ih: Tensor,
    pub dweight_hh: Tensor,
    pub dbias: Tensor,
}

pub fn lstm_cell_backward(
    x: &Tensor,
    h_prev: &Tensor,
    c_prev: &Tensor,
    weight_ih: &Tensor,
    weight_hh: &Tensor,
    gates_raw: &Tensor,
    tanh_c_next: &Tensor,
    dh_next: &Tensor,
    dc_next_upstream: &Tensor,
) -> Result<LstmCellGrads, TensorError> {
    let x_c = x.make_contiguous()?;
    let h_c = h_prev.make_contiguous()?;
    let cp = c_prev.make_contiguous()?;
    let wih = weight_ih.make_contiguous()?;
    let whh = weight_hh.make_contiguous()?;
    let gr = gates_raw.make_contiguous()?;
    let tc = tanh_c_next.make_contiguous()?;
    let dhn = dh_next.make_contiguous()?;
    let dcn = dc_next_upstream.make_contiguous()?;

    let batch = x_c.shape[0];
    let input_size = x_c.shape[1];
    let hidden = h_c.shape[1];

    let mut dx = vec![0.0_f32; batch * input_size];
    let mut dh_prev = vec![0.0_f32; batch * hidden];
    let mut dc_prev = vec![0.0_f32; batch * hidden];
    let mut dweight_ih = vec![0.0_f32; 4 * hidden * input_size];
    let mut dweight_hh = vec![0.0_f32; 4 * hidden * hidden];
    let mut dbias = vec![0.0_f32; 4 * hidden];
    let mut dgates = vec![0.0_f32; batch * 4 * hidden];

    for n in 0..batch {
        for j in 0..hidden {
            let i_raw = gr.data[n * 4 * hidden + j];
            let f_raw = gr.data[n * 4 * hidden + hidden + j];
            let g_raw = gr.data[n * 4 * hidden + 2 * hidden + j];
            let o_raw = gr.data[n * 4 * hidden + 3 * hidden + j];

            let i_gate = sigmoid_f32(i_raw);
            let f_gate = sigmoid_f32(f_raw);
            let g_gate = g_raw.tanh();
            let o_gate = sigmoid_f32(o_raw);

            let dh = dhn.data[n * hidden + j];
            let tc_j = tc.data[n * hidden + j];

            // dc from dh via h = o * tanh(c)
            let dc = dh * o_gate * (1.0 - tc_j * tc_j) + dcn.data[n * hidden + j];

            dc_prev[n * hidden + j] = dc * f_gate;

            // gate gradients (pre-activation)
            dgates[n * 4 * hidden + j]            = dc * g_gate * i_gate * (1.0 - i_gate); // di
            dgates[n * 4 * hidden + hidden + j]   = dc * cp.data[n * hidden + j] * f_gate * (1.0 - f_gate); // df
            dgates[n * 4 * hidden + 2 * hidden + j] = dc * i_gate * (1.0 - g_gate * g_gate); // dg
            dgates[n * 4 * hidden + 3 * hidden + j] = dh * tc_j * o_gate * (1.0 - o_gate); // do
        }

        // Backprop through W_ih and W_hh
        for g in 0..4 * hidden {
            let dg = dgates[n * 4 * hidden + g];
            dbias[g] += dg;
            // dx += W_ih[g, :] * dg
            for k in 0..input_size {
                dx[n * input_size + k] += wih.data[g * input_size + k] * dg;
                dweight_ih[g * input_size + k] += x_c.data[n * input_size + k] * dg;
            }
            // dh_prev += W_hh[g, :] * dg
            for k in 0..hidden {
                dh_prev[n * hidden + k] += whh.data[g * hidden + k] * dg;
                dweight_hh[g * hidden + k] += h_c.data[n * hidden + k] * dg;
            }
        }
    }

    Ok(LstmCellGrads {
        dx: Tensor::new(vec![batch, input_size], dx)?,
        dh_prev: Tensor::new(vec![batch, hidden], dh_prev)?,
        dc_prev: Tensor::new(vec![batch, hidden], dc_prev)?,
        dweight_ih: Tensor::new(vec![4 * hidden, input_size], dweight_ih)?,
        dweight_hh: Tensor::new(vec![4 * hidden, hidden], dweight_hh)?,
        dbias: Tensor::new(vec![4 * hidden], dbias)?,
    })
}

// ─── GRU cell ─────────────────────────────────────────────────────────────────
// weight_ih: [3*hidden, input_size] — rows: [z, r, n] gates for input
// weight_hh: [3*hidden, hidden]     — rows: [z, r, n] gates for h_prev
// bias_ih: [3*hidden], bias_hh: [3*hidden]

pub struct GruCellOutput {
    pub h_next: Tensor,
    /// Saved for backward: z gate [batch, hidden], r gate [batch, hidden], n gate [batch, hidden]
    pub z_gate: Tensor,
    pub r_gate: Tensor,
    pub n_gate: Tensor,
    /// n_raw before tanh [batch, hidden]
    pub n_raw: Tensor,
}

pub fn gru_cell(
    x: &Tensor,
    h_prev: &Tensor,
    weight_ih: &Tensor,
    weight_hh: &Tensor,
    bias_ih: &Tensor,
    bias_hh: &Tensor,
) -> Result<GruCellOutput, TensorError> {
    let x_c = x.make_contiguous()?;
    let h_c = h_prev.make_contiguous()?;
    let wih = weight_ih.make_contiguous()?;
    let whh = weight_hh.make_contiguous()?;
    let bih = bias_ih.make_contiguous()?;
    let bhh = bias_hh.make_contiguous()?;

    if x_c.shape.len() != 2 || h_c.shape.len() != 2 {
        return Err(TensorError { message: "GRU cell expects rank-2 x and h_prev".to_string() });
    }
    let batch = x_c.shape[0];
    let input_size = x_c.shape[1];
    let hidden = h_c.shape[1];
    if wih.shape != vec![3 * hidden, input_size] || whh.shape != vec![3 * hidden, hidden] {
        return Err(TensorError { message: "GRU weight shape mismatch".to_string() });
    }

    let mut z_gate = vec![0.0_f32; batch * hidden];
    let mut r_gate = vec![0.0_f32; batch * hidden];
    let mut n_raw = vec![0.0_f32; batch * hidden];
    let mut n_gate = vec![0.0_f32; batch * hidden];
    let mut h_next = vec![0.0_f32; batch * hidden];

    for n in 0..batch {
        // Compute ih = W_ih @ x + bias_ih (3*hidden)
        let mut ih = vec![0.0_f32; 3 * hidden];
        for g in 0..3 * hidden {
            let mut v = bih.data[g] as f64;
            for k in 0..input_size {
                v += wih.data[g * input_size + k] as f64 * x_c.data[n * input_size + k] as f64;
            }
            ih[g] = v as f32;
        }
        // Compute hh = W_hh @ h_prev + bias_hh (3*hidden)
        let mut hh = vec![0.0_f32; 3 * hidden];
        for g in 0..3 * hidden {
            let mut v = bhh.data[g] as f64;
            for k in 0..hidden {
                v += whh.data[g * hidden + k] as f64 * h_c.data[n * hidden + k] as f64;
            }
            hh[g] = v as f32;
        }

        // z gate (update): sigmoid(ih[0:h] + hh[0:h])
        // r gate (reset):  sigmoid(ih[h:2h] + hh[h:2h])
        // n gate (new):    tanh(ih[2h:3h] + r * hh[2h:3h])
        for j in 0..hidden {
            let z = sigmoid_f32(ih[j] + hh[j]);
            let r = sigmoid_f32(ih[hidden + j] + hh[hidden + j]);
            let n_r = ih[2 * hidden + j] + r * hh[2 * hidden + j];
            let n_a = n_r.tanh();
            z_gate[n * hidden + j] = z;
            r_gate[n * hidden + j] = r;
            n_raw[n * hidden + j] = n_r;
            n_gate[n * hidden + j] = n_a;
            h_next[n * hidden + j] = (1.0 - z) * h_c.data[n * hidden + j] + z * n_a;
        }
    }

    Ok(GruCellOutput {
        h_next: Tensor::new(vec![batch, hidden], h_next)?,
        z_gate: Tensor::new(vec![batch, hidden], z_gate)?,
        r_gate: Tensor::new(vec![batch, hidden], r_gate)?,
        n_gate: Tensor::new(vec![batch, hidden], n_gate)?,
        n_raw: Tensor::new(vec![batch, hidden], n_raw)?,
    })
}

pub struct GruCellGrads {
    pub dx: Tensor,
    pub dh_prev: Tensor,
    pub dweight_ih: Tensor,
    pub dweight_hh: Tensor,
    pub dbias_ih: Tensor,
    pub dbias_hh: Tensor,
}

pub fn gru_cell_backward(
    x: &Tensor,
    h_prev: &Tensor,
    weight_ih: &Tensor,
    weight_hh: &Tensor,
    z_gate: &Tensor,
    r_gate: &Tensor,
    n_gate: &Tensor,
    dh_next: &Tensor,
) -> Result<GruCellGrads, TensorError> {
    let x_c = x.make_contiguous()?;
    let h_c = h_prev.make_contiguous()?;
    let wih = weight_ih.make_contiguous()?;
    let whh = weight_hh.make_contiguous()?;
    let z = z_gate.make_contiguous()?;
    let r = r_gate.make_contiguous()?;
    let n = n_gate.make_contiguous()?;
    let dhn = dh_next.make_contiguous()?;

    let batch = x_c.shape[0];
    let input_size = x_c.shape[1];
    let hidden = h_c.shape[1];

    let mut dx = vec![0.0_f32; batch * input_size];
    let mut dh_prev = vec![0.0_f32; batch * hidden];
    let mut dweight_ih = vec![0.0_f32; 3 * hidden * input_size];
    let mut dweight_hh = vec![0.0_f32; 3 * hidden * hidden];
    let mut dbias_ih = vec![0.0_f32; 3 * hidden];
    let mut dbias_hh = vec![0.0_f32; 3 * hidden];

    for b in 0..batch {
        // dh_next gradient at output
        for j in 0..hidden {
            let dh = dhn.data[b * hidden + j];
            let z_j = z.data[b * hidden + j];
            let r_j = r.data[b * hidden + j];
            let n_j = n.data[b * hidden + j];
            let h_j = h_c.data[b * hidden + j];

            // h_next = (1-z)*h_prev + z*n
            // dh_prev += dh * (1-z)
            dh_prev[b * hidden + j] += dh * (1.0 - z_j);

            // dn = dh * z
            let dn = dh * z_j;
            // n = tanh(n_raw) → dn_raw = dn * (1 - n^2)
            let dn_raw = dn * (1.0 - n_j * n_j);

            // dz = dh * (n - h_prev)
            let dz = dh * (n_j - h_j);
            // dz_raw = dz * z*(1-z)
            let dz_raw = dz * z_j * (1.0 - z_j);

            // n_raw = ih[2h+j] + r * hh[2h+j]
            // dr = dn_raw * hh[2h+j]  (need to accumulate hh product)
            // We need hh[2h+j] = W_hh[2h+j, :] @ h_prev
            // Compute it on the fly
            let mut hh_n_j = 0.0_f32;
            for k in 0..hidden {
                hh_n_j += whh.data[(2 * hidden + j) * hidden + k] * h_c.data[b * hidden + k];
            }
            let dr_raw_pre = dn_raw * hh_n_j;
            let dr_raw = dr_raw_pre * r_j * (1.0 - r_j);

            // Gradients for ih: z_raw (gate 0..h), r_raw (h..2h), n_raw (2h..3h)
            dbias_ih[j] += dz_raw;
            dbias_ih[hidden + j] += dr_raw;
            dbias_ih[2 * hidden + j] += dn_raw;

            // Gradients for hh: same structure for z and r; for n: dr * r_j
            dbias_hh[j] += dz_raw;
            dbias_hh[hidden + j] += dr_raw;
            dbias_hh[2 * hidden + j] += dn_raw * r_j;

            // dh_prev also gets contribution from dr through r gate
            // r_raw = ih[h+j] + hh[h+j] → dhh[h+j] feeds back
            // dh_prev from n path: dn_raw * r through W_hh[2h+j, k]
            for k in 0..hidden {
                dh_prev[b * hidden + k] += whh.data[(2 * hidden + j) * hidden + k] * dn_raw * r_j;
                dh_prev[b * hidden + k] += whh.data[(hidden + j) * hidden + k] * dr_raw;
                dh_prev[b * hidden + k] += whh.data[j * hidden + k] * dz_raw;
            }

            // Accumulate weight grads and dx
            for k in 0..input_size {
                dx[b * input_size + k] += wih.data[j * input_size + k] * dz_raw;
                dx[b * input_size + k] += wih.data[(hidden + j) * input_size + k] * dr_raw;
                dx[b * input_size + k] += wih.data[(2 * hidden + j) * input_size + k] * dn_raw;

                dweight_ih[j * input_size + k] += x_c.data[b * input_size + k] * dz_raw;
                dweight_ih[(hidden + j) * input_size + k] += x_c.data[b * input_size + k] * dr_raw;
                dweight_ih[(2 * hidden + j) * input_size + k] += x_c.data[b * input_size + k] * dn_raw;
            }

            for k in 0..hidden {
                dweight_hh[j * hidden + k] += h_c.data[b * hidden + k] * dz_raw;
                dweight_hh[(hidden + j) * hidden + k] += h_c.data[b * hidden + k] * dr_raw;
                dweight_hh[(2 * hidden + j) * hidden + k] += h_c.data[b * hidden + k] * dn_raw * r_j;
            }
        }
    }

    Ok(GruCellGrads {
        dx: Tensor::new(vec![batch, input_size], dx)?,
        dh_prev: Tensor::new(vec![batch, hidden], dh_prev)?,
        dweight_ih: Tensor::new(vec![3 * hidden, input_size], dweight_ih)?,
        dweight_hh: Tensor::new(vec![3 * hidden, hidden], dweight_hh)?,
        dbias_ih: Tensor::new(vec![3 * hidden], dbias_ih)?,
        dbias_hh: Tensor::new(vec![3 * hidden], dbias_hh)?,
    })
}

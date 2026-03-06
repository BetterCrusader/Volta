use crate::ast::{Expr, Program, Stmt};

pub fn emit_pytorch(program: &Program) -> String {
    let mut out = String::new();
    out.push_str("import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\n");

    for stmt in &program.statements {
        match stmt {
            Stmt::Model { name, props, .. } => {
                let mut layers = Vec::new();
                let mut activation = "nn.ReLU()";
                let mut _optimizer = "SGD";
                let mut _lr = "0.01";

                for prop in props {
                    match prop.key.as_str() {
                        "layers" => {
                            for val in &prop.values {
                                if let Expr::Int { value, .. } = val {
                                    layers.push(*value);
                                }
                            }
                        }
                        "activation" => {
                            if let Some(Expr::Ident { name: act_name, .. }) = prop.values.first() {
                                activation = match act_name.as_str() {
                                    "sigmoid" => "nn.Sigmoid()",
                                    "relu" => "nn.ReLU()",
                                    "softmax" => "nn.Softmax(dim=1)",
                                    "gelu" => "nn.GELU()",
                                    _ => "nn.ReLU()",
                                };
                            }
                        }
                        "optimizer" => {
                            if let Some(Expr::Ident { name: opt_name, .. }) = prop.values.first() {
                                _optimizer = match opt_name.as_str() {
                                    "adam" => "Adam",
                                    _ => "SGD",
                                };
                            }
                        }
                        "lr" => {
                            if let Some(Expr::Float { value, .. }) = prop.values.first() {
                                _lr = Box::leak(value.to_string().into_boxed_str());
                            }
                        }
                        _ => {}
                    }
                }

                out.push_str(&format!("class {}(nn.Module):\n", uppercase_first(name)));
                out.push_str("    def __init__(self):\n");
                out.push_str("        super().__init__()\n");

                let mut layer_defs = String::new();
                for i in 0..layers.len().saturating_sub(1) {
                    layer_defs.push_str(&format!(
                        "            nn.Linear({}, {}),\n",
                        layers[i],
                        layers[i + 1]
                    ));
                    if i < layers.len() - 2 {
                        layer_defs.push_str(&format!("            {},\n", activation));
                    }
                }
                out.push_str("        self.model = nn.Sequential(\n");
                out.push_str(&layer_defs);
                out.push_str("        )\n\n");
                out.push_str("    def forward(self, x):\n");
                out.push_str("        return self.model(x)\n\n");
            }
            Stmt::Dataset { name, props, .. } => {
                let mut batch = "32";
                let mut shuffle = "False";
                for prop in props {
                    if prop.key == "batch"
                        && let Some(Expr::Int { value, .. }) = prop.values.first()
                    {
                        batch = Box::leak(value.to_string().into_boxed_str());
                    }
                    if prop.key == "shuffle"
                        && let Some(Expr::Bool { value, .. }) = prop.values.first()
                    {
                        shuffle = if *value { "True" } else { "False" };
                    }
                }
                out.push_str(&format!("# Placeholder for dataset loading: {}\n", name));
                out.push_str(&format!("{}_loader = DataLoader(TensorDataset(torch.randn(100, 2), torch.randn(100, 1)), batch_size={}, shuffle={})\n\n", name, batch, shuffle));
            }
            Stmt::Train {
                model, data, props, ..
            } => {
                let mut epochs = "10";
                let mut device = "cpu";
                for prop in props {
                    if prop.key == "epochs"
                        && let Some(Expr::Int { value, .. }) = prop.values.first()
                    {
                        epochs = Box::leak(value.to_string().into_boxed_str());
                    }
                    if prop.key == "device"
                        && let Some(Expr::Ident { name: dev_name, .. }) = prop.values.first()
                    {
                        device = Box::leak(dev_name.to_string().into_boxed_str());
                    }
                }
                out.push_str(&format!("device = torch.device('cuda' if '{}' == 'cuda' and torch.cuda.is_available() else 'cpu')\n", device));
                out.push_str(&format!(
                    "model = {}().to(device)\n",
                    uppercase_first(model)
                ));
                out.push_str("optimizer = optim.SGD(model.parameters(), lr=0.01) # Use actual lr/opt from model decl\n");
                out.push_str("criterion = nn.MSELoss()\n\n");
                out.push_str(&format!("for epoch in range({}):\n", epochs));
                out.push_str("    for batch_x, batch_y in ");
                out.push_str(data);
                out.push_str("_loader:\n");
                out.push_str("        batch_x, batch_y = batch_x.to(device), batch_y.to(device)\n");
                out.push_str("        optimizer.zero_grad()\n");
                out.push_str("        outputs = model(batch_x)\n");
                out.push_str("        loss = criterion(outputs, batch_y)\n");
                out.push_str("        loss.backward()\n");
                out.push_str("        optimizer.step()\n");
                out.push_str("    print(f'Epoch {epoch+1} completed. Loss: {loss.item()}')\n\n");
            }
            _ => {}
        }
    }

    out
}

fn uppercase_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

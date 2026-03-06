import torch
import torch.nn as nn
import time

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(512, 1024), nn.ReLU(),
    nn.Linear(1024, 1024), nn.ReLU(),
    nn.Linear(1024, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 1),
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(64, 512)
y = torch.randn(64, 1)

start = time.time()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
elapsed = time.time() - start
print(f"PyTorch: {elapsed:.3f}s for 50 epochs (512->1024->1024->512->256->1, batch=64)")
print(f"Threads: {torch.get_num_threads()}")

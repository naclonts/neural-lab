#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
docker compose exec -T lab python - <<'PY'
import os
import tempfile
import torch

assert os.geteuid() != 0, "Worker must run as a non-root user"
with open("/proc/mounts", encoding="utf-8") as mounts:
    root_options = next(line.split()[3].split(",") for line in mounts if line.split()[1] == "/")
assert "ro" in root_options, "Container root filesystem must be read-only"
for directory in ("/workspace", "/tmp"):
    with tempfile.TemporaryFile(dir=directory) as writable:
        writable.write(b"neural-lab smoke test")

assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver and Container Toolkit"
device = torch.device("cuda:0")
model = torch.nn.Sequential(
    torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
inputs = torch.randn(32, 16, device=device)
targets = torch.randint(0, 2, (32,), device=device)
loss = torch.nn.functional.cross_entropy(model(inputs), targets)
assert torch.isfinite(loss).item(), "Non-finite training loss"
optimizer.zero_grad()
loss.backward()
optimizer.step()
torch.cuda.synchronize()
print("PASS: non-root user, read-only root, writable workspace/tmp, and GPU training")
print("GPU:", torch.cuda.get_device_name(0))
PY

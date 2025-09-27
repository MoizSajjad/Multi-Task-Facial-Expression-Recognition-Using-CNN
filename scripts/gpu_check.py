import torch, numpy as np, sys
print("python:", sys.executable)
print("torch cuda available:", torch.cuda.is_available())
print("torch build cuda:", torch.version.cuda)
print("numpy:", np.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(4096,4096, device="cuda") @ torch.randn(4096,4096, device="cuda")
    print("matmul device:", x.device)
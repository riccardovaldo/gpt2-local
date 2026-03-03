import torch

print(f"CUDA status: {torch.cuda.is_available()}")

print(f"Current GPU: {torch.cuda.get_device_name(0)}")

allocated = torch.cuda.memory_allocated(0) / 1024**2
print(f"Memory used on GPU: {allocated:.2f} MB")
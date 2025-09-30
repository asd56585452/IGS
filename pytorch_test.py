import torch

print("PyTorch 版本 (Version):", torch.__version__)
print("PyTorch 編譯所用的 CUDA 版本 (CUDA Version):", torch.version.cuda)
print("CUDA 是否可用 (CUDA Available):", torch.cuda.is_available())

if torch.cuda.is_available():
    print("當前 GPU 名稱 (Device Name):", torch.cuda.get_device_name(0))
    print("當前 GPU 計算能力 (Compute Capability):", torch.cuda.get_device_capability(0))
else:
    print("找不到 CUDA 裝置。")
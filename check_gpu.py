import torch

if torch.cuda.is_available():
    print("CUDA is available! Your GPU is ready for use.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. No GPU found or the driver is not installed.")

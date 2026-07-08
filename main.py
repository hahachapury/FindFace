import torch
import os
print(os.environ['PATH'])
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
print(torch.__version__)
print(torch.cuda.is_available())  # Должно вернуть True
print(torch.cuda.get_device_name(0)) # Имя вашей GPU
print(torch.version.cuda) 

import torch
import torchvision
print(torch.__version__, "\n", torchvision.__version__)

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import cv2
print("OpenCV version:", cv2.__version__)
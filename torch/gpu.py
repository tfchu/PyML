import torch
import torch.nn as nn
import torch.nn.functional as F

cpu = torch.device("cpu")
gpu = torch.device("cuda")  # cuda:0 (1st GPU), cuda:1 (2nd GPU)

x = torch.rand(10)
print(x)
x = x.to(gpu)
print(x)
x = x.to(cpu)
print(x)

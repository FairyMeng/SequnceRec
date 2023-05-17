from torch import nn
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

model = nn.Linear(2,1,bias=False)
input = torch.Tensor([1,2])
w = torch.normal(0,0.01,size=(1,2),requires_grad=False)
print(w)

output = model(input)
print(output)
import numpy as np
import torch
f = [[2, 2, 2], [2, 2, 2]]
d = [[1, 2, 3], [2, 4, 5], [2, 2, 3]]
f = torch.tensor(f, dtype=torch.float32)
d = torch.tensor(d, dtype=torch.float32)
print(f)
F = torch.norm(f,keepdim=True)
D = torch.norm(d, p=2, keepdim=True)
print(F.pow(2), D)
# print(F+D)

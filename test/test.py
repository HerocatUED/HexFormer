import torch

d = 'cpu'
b = torch.randint(2, (5,1), device=d)
k = torch.randint(4, (5,1), device=d)
z = torch.concatenate((b, k), axis=-1)
print(b)
print(k)
print(z)

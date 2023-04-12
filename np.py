import numpy as np
import torch

a = np.fromfile("emb[688].raw", dtype=np.float32)
t = torch.from_numpy(a)

print("f32 numpy", a.shape, f"{a.std()}")
print("f32 torch", t.std(unbiased=False).numpy())

a=np.float64(a)
t = torch.from_numpy(a)

print("f64 numpy", a.shape, f"{a.std()}")
print("f64 torch", t.std(unbiased=False).numpy())


import numpy as np

a = np.fromfile("emb[688].raw", dtype=np.float32)

print(a.shape, f"{a.std()}")

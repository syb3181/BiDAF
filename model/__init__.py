# Error:
# from_pretrain doesnt modify weight matrix inplace.

import torch
from torch.nn import Embedding

import numpy as np

a = Embedding(num_embeddings=5, embedding_dim=3, padding_idx=1)

z = np.ones((5, 3), dtype=float)
print(z)

z = torch.from_numpy(z)
print(z)

a.from_pretrained(z)
print(a.weight)
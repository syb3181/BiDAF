import numpy as np

w = np.zeros((3, 5), dtype=float)
np.savez('../output/w.npz', w=w)

z = np.load('../output/w.npz')
print(z['w'])

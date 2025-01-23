import numpy as np
import torch
from tmm_fast import coh_tmm as tmm

torch.manual_seed(0)
np.random.seed(0)

wl = np.linspace(400, 1200, 800) * (10**(-6))
theta = np.linspace(0, 0, 1) * (np.pi/180)
mode = 'T'
num_layers = 4
num_stacks = 1

#create m
M = np.ones((num_stacks, num_layers, wl.shape[0]))
for i in range(1, M.shape[1]-1):
    if np.mod(i, 2) == 1:
        M[:, i, :] *= 1.46
    else:
        M[:, i, :] *= 2.56

#create t
max_t = 150 * (10**(-6))
min_t = 10 * (10**(-6))
T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t
T[:, 0] = np.inf
T[:, -1] = np.inf

#tmm:
d = tmm('s', torch.tensor(M), torch.tensor(T), torch.tensor(theta), torch.tensor(wl), device='cpu')
import matplotlib.pyplot as plt
plt.plot(wl, d['R'].T.squeeze())
plt.show()
print(d['R'])
print(':)')
print("ende")
print(d)


import numpy as np
import matplotlib.pyplot as plt
import torch
import jax
import jax.numpy as jnp
from tmm_fast import coh_tmm, coh_tmm_jax

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

print("Creating test case for TMM comparison...")

# Define the wavelength range and incident angle
wl = np.linspace(400, 1200, 800) * (10**(-6))  # Convert to meters
theta = np.linspace(0, 0, 1) * (np.pi/180)  # Normal incidence
mode = 'T'
num_layers = 4
num_stacks = 1

# Create refractive index matrix
M = np.ones((num_stacks, num_layers, wl.shape[0]))
for i in range(1, M.shape[1]-1):
    if np.mod(i, 2) == 1:
        M[:, i, :] *= 1.46  # SiO2-like
    else:
        M[:, i, :] *= 2.56  # TiO2-like

# Create thickness matrix
max_t = 150 * (10**(-6))  # 150 μm
min_t = 10 * (10**(-6))   # 10 μm
T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t
T[:, 0] = np.inf  # Infinite thickness for first layer (substrate)
T[:, -1] = np.inf  # Infinite thickness for last layer (air)

print("Running PyTorch TMM calculation...")
# PyTorch TMM calculation
d_torch = coh_tmm('s', torch.tensor(M), torch.tensor(T), torch.tensor(theta), torch.tensor(wl), device='cpu')

print("Running JAX TMM calculation...")
# JAX TMM calculation
d_jax = coh_tmm_jax('s', np.array(M), np.array(T), np.array(theta), np.array(wl))

# Convert results to numpy for comparison
R_torch = d_torch['R'].cpu().numpy()
T_torch = d_torch['T'].cpu().numpy()
r_torch = d_torch['r'].cpu().numpy()
t_torch = d_torch['t'].cpu().numpy()

R_jax = np.array(d_jax['R'])
T_jax = np.array(d_jax['T'])
r_jax = np.array(d_jax['r'])
t_jax = np.array(d_jax['t'])

# Calculate differences
R_diff = np.abs(R_torch - R_jax)
T_diff = np.abs(T_torch - T_jax)
r_diff = np.abs(r_torch - r_jax)
t_diff = np.abs(t_torch - t_jax)

print(f"Maximum difference in R: {np.max(R_diff)}")
print(f"Maximum difference in T: {np.max(T_diff)}")
print(f"Maximum difference in r: {np.max(r_diff)}")
print(f"Maximum difference in t: {np.max(t_diff)}")

# Plot results for comparison
plt.figure(figsize=(12, 8))

# Plot R
plt.subplot(2, 2, 1)
plt.plot(wl * 1e6, R_torch.squeeze(), 'b-', label='PyTorch')
plt.plot(wl * 1e6, R_jax.squeeze(), 'r--', label='JAX')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Reflectivity (R)')
plt.title('Reflectivity Comparison')
plt.legend()

# Plot T
plt.subplot(2, 2, 2)
plt.plot(wl * 1e6, T_torch.squeeze(), 'b-', label='PyTorch')
plt.plot(wl * 1e6, T_jax.squeeze(), 'r--', label='JAX')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Transmissivity (T)')
plt.title('Transmissivity Comparison')
plt.legend()

# Plot differences
plt.subplot(2, 2, 3)
plt.semilogy(wl * 1e6, R_diff.squeeze(), 'g-', label='R Difference')
plt.semilogy(wl * 1e6, T_diff.squeeze(), 'm-', label='T Difference')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Absolute Difference (log scale)')
plt.title('Differences in R and T')
plt.legend()

# Plot phase differences
plt.subplot(2, 2, 4)
plt.semilogy(wl * 1e6, r_diff.squeeze(), 'g-', label='r Difference')
plt.semilogy(wl * 1e6, t_diff.squeeze(), 'm-', label='t Difference')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Absolute Difference (log scale)')
plt.title('Differences in r and t')
plt.legend()

plt.tight_layout()
plt.savefig('tmm_comparison.png')
plt.show()

print("Comparison complete. Results saved to 'tmm_comparison.png'")
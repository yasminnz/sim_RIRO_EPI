import numpy as np
import nibabel as nib
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# -----------------------------------------------------------------------------#
# Load images
# -----------------------------------------------------------------------------#
sc_mask = nib.load('/spinal_cord_seg.nii')
sim_img = nib.load('/GRE-T1w.nii')

sim_img_data = sim_img.get_fdata()
sc_mask_data = sc_mask.get_fdata()
sc_mask_data[sc_mask_data < 1] = 0

# -----------------------------------------------------------------------------#
# Image acquisition parameters
# -----------------------------------------------------------------------------#
matrix = sim_img.shape
image_res = sim_img.header.get_zooms()
fov = np.array(image_res) * np.array(matrix)

TE = 40e-3  # [s], time to center of k-space
readout_time_per_line = 1e-3  # [s], echo spacing per ky line
readout_time_per_kx = readout_time_per_line / matrix[0]  # Time per kx point (e.g., 6.25 µs for 128 points)
readout_time = readout_time_per_line * matrix[1]  # Total readout time
readout_start = TE - (readout_time / 2)  # Start time so TE is at center
TR = 1000e-3  # [s], for multi-shot (not used here)

# -----------------------------------------------------------------------------#
# Define k-space constants for EPI trajectory
# -----------------------------------------------------------------------------#
k_max = 1 / (2 * np.array(image_res))
delta_k = 1 / fov

kx = np.linspace(-k_max[0], k_max[0], matrix[0])  # Frequency encode
ky = np.linspace(-k_max[1], k_max[1], matrix[1])  # Phase encode
time_per_ky = np.linspace(readout_start, readout_start + readout_time, matrix[1])  # One time per ky line

k_traj = np.zeros((matrix[0] * matrix[1], 2))
for i in range(matrix[1]):
    start_idx = i * matrix[0]
    end_idx = (i + 1) * matrix[0]
    if i % 2 == 0:  # Forward kx
        k_traj[start_idx:end_idx, 0] = kx
    else:  # Reverse kx
        k_traj[start_idx:end_idx, 0] = kx[::-1]
    k_traj[start_idx:end_idx, 1] = ky[i]

# -----------------------------------------------------------------------------#
# Define other constants
# -----------------------------------------------------------------------------#
w_r = 2 * math.pi * 5 / 3
RIROmax_uniform = 100

# -----------------------------------------------------------------------------#
# Define the spatial distribution for RIROmax
# -----------------------------------------------------------------------------#
[x, y] = np.meshgrid(np.linspace(-(matrix[0]-1)/2, (matrix[0]-1)/2, matrix[0]),
                     np.linspace(-(matrix[1]-1)/2, (matrix[1]-1)/2, matrix[1]), indexing='ij')
r = np.sqrt((x * image_res[0])**2 + (y * image_res[1])**2)
r = abs((r - np.max(r)) / np.max(r))
r = r**4
sim_RIROmax = RIROmax_uniform * r

# -----------------------------------------------------------------------------#
# Limit RIROmax to signal areas
# -----------------------------------------------------------------------------#
noise_mask = np.zeros(matrix)
noise_mask[0:4, 0:4] = 1
noise_mask[0:4, (-1-4):-1] = 1
noise_mask[(-1-4):-1, 0:4] = 1
noise_mask[(-1-4):-1, (-1-4):-1] = 1

noise_data = np.multiply(sim_img_data, noise_mask)
sigma = noise_data[noise_data != 0].std()

bkgrnd_mask = np.zeros(matrix)
bkgrnd_mask[sim_img_data > (15 * sigma)] = 1
sim_RIROmax = np.multiply(bkgrnd_mask, sim_RIROmax)

mean_simRIROmax = sim_RIROmax[bkgrnd_mask != 0].mean()
std_simRIROmax = sim_RIROmax[bkgrnd_mask != 0].std()
print('sim RIROmax mean =', mean_simRIROmax)
print('sim RIROmax std =', std_simRIROmax)

# -----------------------------------------------------------------------------#
# Simulate EPI k-space acquisition with RIROmax (per ky line)
# -----------------------------------------------------------------------------#
# Ideal image in k-space
sim_FFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sim_img_data)))

# Preallocate k-space signal along trajectory
k_space_mod_RIROmax = np.zeros(matrix[0] * matrix[1], dtype=complex)

# Compute phase and sample k-space per ky line
for ky_idx, t in enumerate(time_per_ky):
    # Phase across all (x, y) for this ky line's time
    phase = (-2 * math.pi * 1j * TE) * sim_RIROmax * np.sin(w_r * t)
    img_with_phase = sim_img_data * np.exp(phase)
    # Fourier transform to k-space
    k_space_t = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_with_phase)))
    # Sample along kx for this ky
    start_idx = ky_idx * matrix[0]
    end_idx = (ky_idx + 1) * matrix[0]
    ky_val = ky[ky_idx]
    ky_t_idx = np.argmin(np.abs(ky - ky_val))
    if ky_idx % 2 == 0:  # Forward kx
        k_space_mod_RIROmax[start_idx:end_idx] = k_space_t[:, ky_t_idx]
    else:  # Reverse kx
        k_space_mod_RIROmax[start_idx:end_idx] = k_space_t[::-1, ky_t_idx]

# Regrid to 2D k-space
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy')
k_space_mod_2D = griddata(k_traj, k_space_mod_RIROmax, (kx_grid, ky_grid), method='cubic', fill_value=0)

# Reconstruct the image
calcImage_sim_RIROmax_xy = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k_space_mod_2D.T)))

# Average RIROmax in spinal cord
sim_RIROmax_meas = np.multiply(sim_RIROmax, sc_mask_data)
sim_RIROmax_meas = sim_RIROmax_meas[np.nonzero(sim_RIROmax_meas)].mean()

# -----------------------------------------------------------------------------#
# Z-shim simulation
# -----------------------------------------------------------------------------#
k_space_zshim = np.zeros(matrix[0] * matrix[1], dtype=complex)

for ky_idx, t in enumerate(time_per_ky):
    diff = sim_RIROmax * np.sin(w_r * t) - sim_RIROmax_meas * np.sin(w_r * t)
    zshim_phase = (2 * math.pi * 1j * TE) * diff
    img_with_zshim = sim_img_data * np.exp(zshim_phase)
    k_space_t = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_with_zshim)))
    start_idx = ky_idx * matrix[0]
    end_idx = (ky_idx + 1) * matrix[0]
    ky_val = ky[ky_idx]
    ky_t_idx = np.argmin(np.abs(ky - ky_val))
    if ky_idx % 2 == 0:  # Forward kx
        k_space_zshim[start_idx:end_idx] = k_space_t[:, ky_t_idx]
    else:  # Reverse kx
        k_space_zshim[start_idx:end_idx] = k_space_t[::-1, ky_t_idx]

# Regrid to 2D k-space
zshim_k_space_2D = griddata(k_traj, k_space_zshim, (kx_grid, ky_grid), method='cubic', fill_value=0)

# Reconstruct the image
zshim_calcImage_sim_RIROmax_xy = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(zshim_k_space_2D.T)))

# -----------------------------------------------------------------------------#
# Display
# -----------------------------------------------------------------------------#
fig = plt.figure(figsize=(25, 15))

ax1 = fig.add_subplot(1, 4, 1)
im1 = ax1.imshow(np.rot90(sim_RIROmax), cmap='jet', vmin=0, vmax=RIROmax_uniform)
ax1.set_title('RIROmax [Hz]')
plt.setp(ax1, xticks=[], yticks=[])
plt.colorbar(im1, fraction=0.046, pad=0.05)

ax2 = fig.add_subplot(1, 4, 2)
ax2.imshow(np.rot90(sim_img_data), cmap='gray', vmin=0, vmax=1200)
ax2.set_title('Ideal Image')
plt.setp(ax2, xticks=[], yticks=[])

ax3 = fig.add_subplot(1, 4, 3)
ax3.imshow(np.rot90(np.abs(calcImage_sim_RIROmax_xy)), cmap='gray', vmin=0, vmax=1200)
ax3.set_title('Simulated EPI measurement')
plt.setp(ax3, xticks=[], yticks=[])

ax4 = fig.add_subplot(1, 4, 4)
ax4.imshow(np.rot90(np.abs(zshim_calcImage_sim_RIROmax_xy)), cmap='gray', vmin=0, vmax=1200)
ax4.set_title('Simulated EPI z-shim measurement')
plt.setp(ax4, xticks=[], yticks=[])

plt.show()

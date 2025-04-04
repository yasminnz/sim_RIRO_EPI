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

TR = 1000e-3  # [s], repetition time
num_TE = 8  # Number of echoes
TE1 = 2.5e-3  # First echo time
delta_TE = 3e-3  # Inter-echo spacing
TEs = np.array([TE1 + i * delta_TE for i in range(num_TE)])  # Echo times

# -----------------------------------------------------------------------------#
# Define k-space constants
# -----------------------------------------------------------------------------#
k_max = 1 / (2 * np.array(image_res))
kx = np.linspace(-k_max[0], k_max[0], matrix[0])
ky = np.linspace(-k_max[1], k_max[1], matrix[1])
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

# Define k-space trajectory for mGRE (one line per TR)
k_traj = np.zeros((matrix[0] * matrix[1], 2))
for i in range(matrix[1]):
    start_idx = i * matrix[0]
    end_idx = (i + 1) * matrix[0]
    k_traj[start_idx:end_idx, 0] = kx
    k_traj[start_idx:end_idx, 1] = ky[i]

# -----------------------------------------------------------------------------#
# Define RIROmax spatial distribution
# -----------------------------------------------------------------------------#
[x, y] = np.meshgrid(np.linspace(-(matrix[0]-1)/2, (matrix[0]-1)/2, matrix[0]),
                     np.linspace(-(matrix[1]-1)/2, (matrix[1]-1)/2, matrix[1]), indexing='ij')
r = np.sqrt((x * image_res[0])**2 + (y * image_res[1])**2)
r = abs((r - np.max(r)) / np.max(r)) ** 4
RIROmax_uniform = 12
sim_RIROmax = RIROmax_uniform * r

# Apply mask to remove background effects
sim_RIROmax *= sc_mask_data

# -----------------------------------------------------------------------------#
# Simulate mGRE k-space acquisition with RIROmax
# -----------------------------------------------------------------------------#
w_r = 2 * math.pi / 3
sim_FFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sim_img_data)))

# Allocate space for multi-echo k-space
k_space_mod_RIROmax = np.zeros((num_TE, matrix[0] * matrix[1]), dtype=complex)

# Loop through each echo and simulate the acquisition
for j, TE in enumerate(TEs):
    for t_idx in range(matrix[0] * matrix[1]):
        # Compute phase modulation due to respiration
        phase = (-2 * math.pi * 1j * TE) * sim_RIROmax * np.sin(w_r * t_idx)
        # Apply phase to ideal image
        img_with_phase = sim_img_data * np.exp(phase)
        # Fourier transform to get k-space at this time
        k_space_t = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_with_phase)))
        # Sample at current [kx, ky] point
        kx_t, ky_t = k_traj[t_idx]
        kx_idx = np.argmin(np.abs(kx - kx_t))
        ky_idx = np.argmin(np.abs(ky - ky_t))
        k_space_mod_RIROmax[j, t_idx] = k_space_t[kx_idx, ky_idx]

# Reshape into 2D k-space for each echo
k_space_reshaped = k_space_mod_RIROmax.reshape((num_TE, matrix[0], matrix[1]))

# -----------------------------------------------------------------------------#
# Image Reconstruction
# -----------------------------------------------------------------------------#
images_reconstructed = np.zeros((num_TE, matrix[0], matrix[1]))

for j in range(num_TE):
    images_reconstructed[j] = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k_space_reshaped[j]))))

# -----------------------------------------------------------------------------#
# Display Results
# -----------------------------------------------------------------------------#
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for j in range(num_TE):
    ax = axes[j // 4, j % 4]
    ax.imshow(np.flipud(images_reconstructed[j]), cmap='gray')
    ax.set_title(f'Echo {j+1}: TE = {TEs[j] * 1000:.1f} ms')
    ax.axis('off')

plt.tight_layout()
plt.show()

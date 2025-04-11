import numpy as np
import nibabel as nib
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# -----------------------------------------------------------------------------#
# Load images
# -----------------------------------------------------------------------------#
sc_mask = nib.load('/sim_data/spinal_cord_seg.nii')
sim_img = nib.load('/sim_data/GRE-T1w.nii')

sim_img_data = sim_img.get_fdata()
sc_mask_data = sc_mask.get_fdata()
sc_mask_data[sc_mask_data < 1] = 0

# -----------------------------------------------------------------------------#
# Image acquisition parameters
# -----------------------------------------------------------------------------#
matrix = sim_img.shape # Image dimensinos (x,y)
image_res = sim_img.header.get_zooms() # Voxel size (mm per pixel)
fov = np.array(image_res) * np.array(matrix)

readout_time = 50e-3  # [s], time to center of k-space
readout_time_per_line = readout_time/matrix[1]  # [s], time per ky line
dwell_time = readout_time_per_line / matrix[0] # [s], time per (kx,ky) point 
readout_start = TE - (readout_time / 2)  # Start time so TE is at center
TR = 1000e-3  # [s], for multi-shot (not used here)

# -----------------------------------------------------------------------------#
# Define k-space constants for EPI trajectory
# -----------------------------------------------------------------------------#
k_max = 1 / (2 * np.array(image_res))
delta_k = 1 / fov

kx = np.linspace(-k_max[0], k_max[0], matrix[0])  # Frequency encode
ky = np.linspace(-k_max[1], k_max[1], matrix[1])  # Phase encode
time = np.zeros(matrix[0] * matrix[1])
for i in range(matrix[1]): # Loop over ky lines
    start_idx = i * matrix[0] # Start index for this line in the flattened array
    end_idx = (i + 1) * matrix[0] # End index for this line in the flattened array
    line_start = readout_start + i * readout_time_per_line # Compute the readout start time for this line
    time[start_idx:end_idx] = line_start + np.arange(matrix[0]) * dwell_time # Assign point-wise readout times

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
w_r = 2 * math.pi / 3 # Breathing cycle frequency (1 breaths per 3s)
RIROmax_uniform = 12 # Maximum frequency shift in Hz

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
# Simulate EPI k-space acquisition with RIROmax
# -----------------------------------------------------------------------------#
# Ideal image in k-space
sim_FFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sim_img_data)))

# Preallocate k-space signal along trajectory
k_space_mod_RIROmax = np.zeros(matrix[0] * matrix[1], dtype=complex)

# Spatial grid for phase calculation
x_grid, y_grid = np.meshgrid(np.linspace(-(matrix[0]-1)/2, (matrix[0]-1)/2, matrix[0]),
                             np.linspace(-(matrix[1]-1)/2, (matrix[1]-1)/2, matrix[1]), indexing='ij')

for t_idx, t in enumerate(time):
    # Compute phase across all (x, y) for this time point
    phase = (-2 * math.pi * 1j ) * sim_RIROmax * (1/w_r) * (1 - np.cos(w_r * t))
    # Apply phase to ideal image
    img_with_phase = sim_img_data * np.exp(phase)
    # Fourier transform to get k-space at this time
    k_space_t = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_with_phase)))
    # Sample at current [kx, ky] point
    kx_t, ky_t = k_traj[t_idx]
    kx_idx = np.argmin(np.abs(kx - kx_t))
    ky_idx = np.argmin(np.abs(ky - ky_t))
    k_space_mod_RIROmax[t_idx] = k_space_t[kx_idx, ky_idx]

# Reshape k-space data into a 2D array (assuming it follows matrix shape)
k_space_reshaped = k_space_mod_RIROmax.reshape(matrix)

# Reconstruct the image
calcImage_sim_RIROmax_xy = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k_space_reshaped)))



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
ax3.imshow(np.flipud(np.abs(calcImage_sim_RIROmax_xy)), cmap='gray', vmin=0, vmax=1200)
ax3.set_title('Simulated EPI measurement')
plt.setp(ax3, xticks=[], yticks=[])



plt.show()

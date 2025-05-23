import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Load and convert image to float
img = cv2.imread('IMG_3038.jpeg', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)
rows, cols = img.shape

# Frequency grid
u = np.arange(cols) - cols // 2
v = np.arange(rows) - rows // 2
U, V = np.meshgrid(u, v)
D2 = U**2 + V**2

# Shifted FFT
F = fftshift(fft2(img))

# Inverted PSF High-Pass Filter (Gaussian subtraction)
sigma = 30
low_pass = np.exp(-D2 / (2 * sigma**2))
high_pass_psf = 1 - low_pass
G_psf = F * high_pass_psf
img_psf = np.real(ifft2(ifftshift(G_psf)))

# Sharp High-Pass Filter (small sigma)
sharp_sigma = 5
low_pass_sharp = np.exp(-D2 / (2 * sharp_sigma**2))
high_pass_sharp = 1 - low_pass_sharp
G_sharp = F * high_pass_sharp
img_sharp = np.real(ifft2(ifftshift(G_sharp)))

# Laplacian High-Pass Filter
H_lap = -(U**2 + V**2).astype(np.float64)
H_lap /= np.max(np.abs(H_lap))  # Normalize
G_lap = F * H_lap
img_lap = np.real(ifft2(ifftshift(G_lap)))

# Plot results
plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.clip(img_psf, 0, 255), cmap='gray')
plt.title('Inverted PSF High Pass')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.clip(img_sharp, 0, 255), cmap='gray')
plt.title('Sharp High Pass')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(np.clip(np.abs(10 * img_lap), 0, 255), cmap='gray')  # Boosted for visibility
plt.title('Laplacian High Pass')
plt.axis('off')

plt.tight_layout()
plt.show()

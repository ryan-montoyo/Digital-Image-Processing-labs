import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Load image and convert to float
img = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)

# Get image size
rows, cols = img.shape

# FFT and shift
F = fftshift(fft2(img))

# Cutoff and Butterworth order
D0 = 30
n = 2

# Build filter grid with same shape as image
u = np.arange(cols) - cols // 2
v = np.arange(rows) - rows // 2
U, V = np.meshgrid(u, v)
D = np.sqrt(U**2 + V**2)

# Ideal Low Pass
H_ideal = np.double(D <= D0)
G_ideal = H_ideal * F
img_ideal = np.real(ifft2(ifftshift(G_ideal)))

# Gaussian Low Pass
H_gaussian = np.exp(-(D**2) / (2 * D0**2))
G_gaussian = H_gaussian * F
img_gaussian = np.real(ifft2(ifftshift(G_gaussian)))

# Butterworth Low Pass
H_butterworth = 1 / (1 + (D / D0)**(2 * n))
G_butterworth = H_butterworth * F
img_butterworth = np.real(ifft2(ifftshift(G_butterworth)))

# Plot results
plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img_ideal, cmap='gray')
plt.title('Ideal Low Pass')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Gaussian Low Pass')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img_butterworth, cmap='gray')
plt.title('Butterworth Low Pass')
plt.axis('off')

plt.tight_layout()
plt.show()


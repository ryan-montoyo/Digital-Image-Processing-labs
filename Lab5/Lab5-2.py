import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale OCT image
img = cv2.imread('Lab5/8bit-image.png', cv2.IMREAD_GRAYSCALE)


# Extract all bitplanes
bitplanes = [(img >> i) & 1 for i in range(8)]

# Zero out noisy bitplanes (0-3)
for i in range(4):
    bitplanes[i] = np.zeros_like(img, dtype=np.uint8)

# Reconstruct image from filtered bitplanes
reconstructed = np.zeros_like(img, dtype=np.uint8)
for i in range(8):
    reconstructed += (bitplanes[i] << i)

# Step 3: Reconstruct image from higher bitplanes
reconstructed = np.zeros_like(img, dtype=np.uint8)
for i in range(8):
    reconstructed += (bitplanes[i] << i)

# Apply averaging filter 
kernel_size = 3  # 3x3 kernel
denoised = cv2.blur(reconstructed, (kernel_size, kernel_size))

# Step 4: Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Reconstructed (bitplanes 4-7)")
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Denoised (Averaging Filter)")
plt.imshow(denoised, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

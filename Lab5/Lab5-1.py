import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
img = cv2.imread('Lab5/8bit-image.png', cv2.IMREAD_GRAYSCALE)


# Create 8 subplots for each bitplane
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Bitplane Decomposition', fontsize=16)

for i in range(8):
    bitplane = (img >> i) & 1  # Extract the i-th bit
    bitplane_img = bitplane * 255  # Scale to 0 or 255 for visualization

    ax = axes[i // 4, i % 4]
    ax.imshow(bitplane_img, cmap='gray')
    ax.set_title(f'Bitplane {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()

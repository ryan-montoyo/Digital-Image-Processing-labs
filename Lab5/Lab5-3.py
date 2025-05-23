import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale MRI image
img = cv2.imread('Lab5/tumor.png', cv2.IMREAD_GRAYSCALE)

# Decompose into 8 bitplanes
bitplanes = [(img >> i) & 1 for i in range(8)]

# Extract 8th bitplane (bitplanes[7]) and scale to 0 or 255
bitplane8 = bitplanes[7] * 255

# Apply 5x5 averaging filter
filtered = cv2.blur(bitplane8, (5, 5)) 


# Apply binary threshold 
_, binary_mask = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)


# Multiply binary mask with original image to extract tumor
tumor_region = cv2.bitwise_and(img, binary_mask)


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title("Original MRI")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Filtered 8th Bitplane")
plt.imshow(filtered, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Thresholded Mask")
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Extracted Tumor Region")
plt.imshow(tumor_region, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

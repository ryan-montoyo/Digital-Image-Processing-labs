import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load both images in grayscale mode
I1 = cv2.imread("wafer.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
I2 = cv2.imread("wafer_defect.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Add the images and clip pixel values to stay within [0, 255]
Isub = cv2.absdiff(I2, I1)

# Display images
plt.subplot(1, 3, 1)
plt.imshow(I1, cmap="gray")
plt.title("Image 1: Template")

plt.subplot(1, 3, 2)
plt.imshow(I2, cmap="gray")
plt.title("Image 2: Defective Wafer")

plt.subplot(1, 3, 3)
plt.imshow(Isub, cmap="gray")
plt.title("Defects")

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load both images in grayscale mode
I1 = cv2.imread("plane.png")  # Load as grayscale
I2 = cv2.imread("car.png")  # Load as grayscale

# Add the images and clip pixel values to stay within [0, 255]
Iadd = np.clip(I1/2 + I2/2, 0, 255).astype(np.uint8)

# Display images
plt.subplot(1, 3, 1)
plt.imshow(I1, cmap="gray")
plt.title("Image 1: Plane")

plt.subplot(1, 3, 2)
plt.imshow(I2, cmap="gray")
plt.title("Image 2: Car")

plt.subplot(1, 3, 3)
plt.imshow(Iadd)
plt.title("Added Image")

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load both images in grayscale mode
I1 = cv2.imread("plane.png")  # Load as grayscale
I2 = cv2.imread("car.png")  # Load as grayscale

# Add the images and clip pixel values to stay within [0, 255]
Iadd1 = np.clip(I1*0.2 + I2*(1-0.2), 0, 255).astype(np.uint8)
Iadd2 = np.clip(I1*0.5 + I2*(1-0.5), 0, 255).astype(np.uint8)
Iadd3 = np.clip(I1*0.8 + I2*(1-0.8), 0, 255).astype(np.uint8)

# Display images
plt.subplot(1, 3, 1)
plt.imshow(Iadd1, cmap="gray")
plt.title("alpha = 0.2")

plt.subplot(1, 3, 2)
plt.imshow(Iadd2, cmap="gray")
plt.title("alpha = 0.5")

plt.subplot(1, 3, 3)
plt.imshow(Iadd3)
plt.title("alpha = 0.8")


plt.show()
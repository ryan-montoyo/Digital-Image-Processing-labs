import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load both images in grayscale mode
I1 = cv2.imread("plane.png", cv2.IMREAD_GRAYSCALE) 
I2 = cv2.imread("car.png", cv2.IMREAD_GRAYSCALE)  


# Add the images and clip pixel values to stay within [0, 255]
Iadd = np.clip(I1.astype(np.int16) + I2.astype(np.int16), 0, 255).astype(np.uint8)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(I1, cmap="gray")  
plt.title("Image 1: Plane")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(I2, cmap="gray")  
plt.title("Image 2: Car")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(Iadd, cmap="gray") 
plt.title("Added Image")
plt.axis("off")

plt.show()


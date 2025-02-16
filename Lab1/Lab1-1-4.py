import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the color image
image = cv2.imread("dog.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Convert the image from RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Create copies for different saturation levels
saturation_1 = hsv_image.copy()
saturation_2 = hsv_image.copy()
saturation_3 = hsv_image.copy()

# Scale the saturation values (and clip to 0-255)
saturation_1[:, :, 1] = saturation_1[:, :, 1] * 0.01
saturation_2[:, :, 1] = saturation_2[:, :, 1] * 0.26
saturation_3[:, :, 1] = saturation_3[:, :, 1] * 0.46

# Convert back to RGB for display
saturation_1_rgb = cv2.cvtColor(saturation_1, cv2.COLOR_HSV2RGB)
saturation_2_rgb = cv2.cvtColor(saturation_2, cv2.COLOR_HSV2RGB)
saturation_3_rgb = cv2.cvtColor(saturation_3, cv2.COLOR_HSV2RGB)


# Display the versions 
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(saturation_1_rgb)  
plt.title("Saturation =  0.01")

plt.subplot(1, 4, 3)
plt.imshow(saturation_2_rgb)  
plt.title("Saturation =  0.26")

plt.subplot(1, 4, 4)
plt.imshow(saturation_3_rgb)  
plt.title("Saturation =  0.46")

plt.show()
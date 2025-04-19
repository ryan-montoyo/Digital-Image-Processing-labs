import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the image
image = cv2.imread("car.png")  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
image = image.astype(np.uint16)

# Extract RGB channels
R = image[:, :, 0]  # Red channel
G = image[:, :, 1]  # Green channel
B = image[:, :, 2]  # Blue channel


luminosity = (R * 0.2989 + G * 0.5870 + B * 0.1140).astype(np.uint8) 

# JET CONVERSION
jet = cv2.applyColorMap(luminosity, cv2.COLORMAP_JET)
jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)


# RAINBOW CONVERSION
I = luminosity.astype(np.float32)  # Convert to float for calculations
R_new = np.sin((np.pi * I / 255) + 0) * 127 + 128
G_new = np.sin((np.pi * I / 255) + (2 * np.pi / 3)) * 127 + 128
B_new = np.sin((np.pi * I / 255) + (4 * np.pi / 3)) * 127 + 128
rainbow = np.stack([R_new, G_new, B_new], axis=-1).astype(np.uint8)


# THERMAL CONVERSION
R_new = 255 * np.maximum(0, (I - 128) / 127)
G_new = 255 * np.minimum(1, I / 128)
B_new = 255 * np.minimum(1, (128 - I) / 128)
thermal = np.stack([R_new, G_new, B_new], axis=-1).astype(np.uint8)

# Display the result
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(luminosity, cmap="gray")
plt.title("Luminosity Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(jet, cmap="gray")
plt.title("JET")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(rainbow, cmap="gray")
plt.title("Rainbow")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(thermal, cmap="gray")
plt.title("Thermal")
plt.axis("off")



plt.show()

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
averaging = np.clip((R.astype(np.uint16) + G.astype(np.uint16) + B.astype(np.uint16)) // 3, 0, 255).astype(np.uint8)
lightness = (np.max(image, axis=-1) + np.min(image, axis=-1,)) // 2
scotopic = (R * 0.0072 + G * 0.8467 + B * 0.146).astype(np.uint8) 
photopic = (R * 0.2126 + G * 0.7152 + B * 0.0722).astype(np.uint8) 

lightness = lightness.astype(np.uint8)

# Display the result
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(luminosity, cmap="gray") 
plt.title("Luminosity Method")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(averaging, cmap="gray")
plt.title("Averaging Method")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(lightness, cmap="gray")
plt.title("Lightness Method")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(scotopic, cmap="gray")
plt.title("Scotopic Method")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(photopic, cmap="gray")
plt.title("Photopic Method")
plt.axis("off")

plt.show()


import cv2
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread("dog.png")  
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (since OpenCV loads in BGR)

# Convert the image from RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Extract the individual channels
hue_channel = hsv_image[:, :, 0]        # Hue
saturation_channel = hsv_image[:, :, 1] # Saturation
value_channel = hsv_image[:, :, 2]      # Value (brightness)

# Display the grayscale versions of the HSV channels
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(hue_channel, cmap="gray")  # Display Hue in grayscale
plt.title("Hue Channel")

plt.subplot(1, 4, 3)
plt.imshow(saturation_channel, cmap="gray")  # Display Saturation in grayscale
plt.title("Saturation Channel")

plt.subplot(1, 4, 4)
plt.imshow(value_channel, cmap="gray")  # Display Value (Brightness) in grayscale
plt.title("Value Channel")

plt.show()

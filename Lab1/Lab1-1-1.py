import cv2
import matplotlib.pyplot as plt


# Load the color image
image = cv2.imread("dog.png")  # Make sure dog.png is in your working directory
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (OpenCV loads in BGR)

# Extract single channels
red_channel = image[:, :, 0]   # Red channel
green_channel = image[:, :, 1] # Green channel
blue_channel = image[:, :, 2]  # Blue channel

# Display the grayscale versions of the channels
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(red_channel, cmap = "gray")
plt.title("Red Channel")

plt.subplot(1, 4, 3)
plt.imshow(green_channel, cmap = "gray")
plt.title("Green Channel")

plt.subplot(1, 4, 4)
plt.imshow(blue_channel, cmap = "gray")
plt.title("Blue Channel")

plt.show()

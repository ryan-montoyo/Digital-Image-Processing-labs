import cv2
import matplotlib.pyplot as plt


# Load the color image
image = cv2.imread("dog.png")  # Make sure dog.png is in your working directory
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (OpenCV loads in BGR)

# Create copies of the image 
red_channel = image.copy()
green_channel = image.copy()
blue_channel = image.copy()

#Change the G and B to 0 
red_channel[:, :, 1] = 0 # Set Green to 0
red_channel[:, :, 2] = 0 # Set Blue to 0

#Change the R and B to 0 
green_channel[:, :, 0] = 0 # Set Red to 0
green_channel[:, :, 2] = 0 # Set Blue to 0

#Change the R and G to 0 
blue_channel[:, :, 0] = 0 # Set Red to 0
blue_channel[:, :, 1] = 0 # Set Green to 0

# Display the different color versions of the channels
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(red_channel)
plt.title("Red Channel")

plt.subplot(1, 4, 3)
plt.imshow(green_channel)
plt.title("Green Channel")

plt.subplot(1, 4, 4)
plt.imshow(blue_channel)
plt.title("Blue Channel")

plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the dog.png image and convert to grayscale
image = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

# 2D Averaging Filter (3x3)
h_avg = (1/9) * np.array([[1, 1, 1], 
                          [1, 1, 1], 
                          [1, 1, 1]])

# 2D Difference Filter (Edge Detection, 3x3)
h_diff = np.array([[-1, -1, -1], 
                   [-1, 8, -1], 
                   [-1, -1, -1]])

# Manual convolution function for 2D filtering
def manual_convolution_2d(image, kernel):
    # Get image and kernel dimensions
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output = np.zeros_like(image, dtype=float)

    # Pad the image to handle border pixels
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    # Normalize the output to fit in the range [0, 255]
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

# Apply the 2D filters to the image
avg_filtered = manual_convolution_2d(image, h_avg)
diff_filtered = manual_convolution_2d(image, h_diff)

# Plotting the original and filtered images
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Averaging Filter")
plt.imshow(avg_filtered, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Difference Filter (Edge Detection)")
plt.imshow(diff_filtered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

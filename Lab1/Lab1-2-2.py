import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread("dog.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale


n = 100  

# Initialize an empty array to store the sum of noisy images
accumulated_image = np.zeros_like(image, dtype=np.float64)

# Generate and add Gaussian noise to the image 100 times
for i in range(n):
    noise = np.random.normal(loc=0, scale=25, size=image.shape)  # Gaussian noise (mean=0, std=25)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)  # Add noise and clip

    accumulated_image += noisy_image  # Accumulate the noisy images

# Compute the average of the noisy images
denoised_image = (accumulated_image / n).astype(np.uint8)

# Display images
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap="gray")  # Last noisy image generated
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised_image, cmap="gray")  # Averaged image
plt.title("Denoised Image")
plt.axis("off")

plt.show()
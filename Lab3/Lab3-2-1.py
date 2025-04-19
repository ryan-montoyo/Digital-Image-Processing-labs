import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image (dog.png) and convert to grayscale
image = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Add Gaussian Distribution Noise
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_uniform_noise(image):
    noise = np.random.uniform(-50, 50, image.shape).astype(np.int16)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = np.copy(image)
    num_salt = int(np.ceil(salt_prob * image.size))
    num_pepper = int(np.ceil(pepper_prob * image.size))

    # Salt noise (white pixels)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Combine both Gaussian and Salt & Pepper noise
def add_combined_noise(image):
    gaussian_noisy = add_gaussian_noise(image)
    combined_noisy = add_salt_pepper_noise(gaussian_noisy)
    return combined_noisy

# Step 2: Filtering Techniques
def median_filter(image):
    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=np.uint8)
    padded_image = np.pad(image, 1, mode='constant', constant_values=0)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Extract the 3x3 neighborhood
            region = padded_image[i-1:i+2, j-1:j+2].flatten()
            # Find the median value
            median_value = np.median(region)
            # Set the median value in the output image
            output[i-1, j-1] = median_value
            
    return output

def alpha_mean_filter(image):
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(image, -1, kernel)

# Correct Gaussian filter as per professor's code
def gaussian_filter(image):
    gaussian_kernel = (1/16) * np.array([[1, 2, 1], 
                                         [2, 4, 2], 
                                         [1, 2, 1]])
    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=np.float32)
    padded_image = np.pad(image, 1, mode='constant')

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            region = padded_image[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum(region * gaussian_kernel)

    return np.clip(output, 0, 255).astype(np.uint8)

# Step 3: PSNR Calculation
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr

# Step 4: Generate Noisy Image
gaussian_noisy = add_uniform_noise(image)

# Apply three different filters to the noisy image
median_filtered = median_filter(gaussian_noisy)
gaussian_filtered = gaussian_filter(gaussian_noisy)
alpha_mean_filtered = alpha_mean_filter(gaussian_noisy)

# Step 5: Calculate PSNR values
psnr_noisy = calculate_psnr(image, gaussian_noisy)
psnr_median = calculate_psnr(gaussian_noisy, median_filtered)
psnr_gaussian = calculate_psnr(gaussian_noisy, gaussian_filtered)
psnr_alpha_mean = calculate_psnr(gaussian_noisy, alpha_mean_filtered)

print("PSNR Values for Gaussian Noise:")
print(f"Noisy Image to Original: {psnr_noisy:.2f} dB")
print(f"Median Filtered to Noisy: {psnr_median:.2f} dB")
print(f"Gaussian Filtered to Noisy: {psnr_gaussian:.2f} dB")
print(f"Alpha Mean Filtered to Noisy: {psnr_alpha_mean:.2f} dB")

# Step 6: Display Images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Noisy Image")
plt.imshow(gaussian_noisy, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Median Filtered")
plt.imshow(median_filtered, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Gaussian Filtered")
plt.imshow(gaussian_filtered, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Alpha Mean Filtered")
plt.imshow(alpha_mean_filtered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

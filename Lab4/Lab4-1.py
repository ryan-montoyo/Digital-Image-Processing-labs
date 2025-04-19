import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the image 
image = cv2.imread("dog.png", cv2.IMREAD_GRAYSCALE)

# Histogram Equalization
def histogram_equalization(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_normalized = 255 * cdf / cdf[-1] 
    equalized = cdf_normalized[img]
    return equalized.astype(np.uint8)

# Linear Contrast Stretching
def linear_contrast_stretching(img):
    I_min = np.min(img)
    I_max = np.max(img)
    stretched = ((img - I_min) / (I_max - I_min)) * 255
    return stretched.astype(np.uint8)

# Gamma Correction 
def gamma_correction(img, gamma_val=2.2):
    norm_img = img / 255.0
    corrected = np.power(norm_img, 1/gamma_val) * 255
    return np.clip(corrected, 0, 255).astype(np.uint8)

# Power-Law Transformation 
def power_law_transform(img, gamma_val=0.5):
    norm_img = img / 255.0
    transformed = np.power(norm_img, gamma_val) * 255
    return np.clip(transformed, 0, 255).astype(np.uint8)

# Apply transformations
he_img = histogram_equalization(image)
lcs_img = linear_contrast_stretching(image)
gamma_img = gamma_correction(image, gamma_val=2.2)
pl_img = power_law_transform(image, gamma_val=0.5)


# Display all results
titles = [
    "Original",
    "Histogram Equalization",
    "Linear Stretching",
    "Gamma Correction (γ=2.2)",
    "Power-Law Transform (γ=0.5)"
]
images = [image, he_img, lcs_img, gamma_img, pl_img]

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

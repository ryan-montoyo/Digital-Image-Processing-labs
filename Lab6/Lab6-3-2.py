import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and convert to grayscale
img = cv2.imread('water.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Define Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=float)

# Manual convolution function
def manual_convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img, dtype=float)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

# Sobel edge function
def sobel_edge(img):
    gx = manual_convolution(img, sobel_x)
    gy = manual_convolution(img, sobel_y)
    edge = np.sqrt(gx**2 + gy**2)
    return np.clip(edge, 0, 255).astype(np.uint8)

# Step (a): 5x5 Median + Sobel
median_5 = cv2.medianBlur(gray, 5)
sobel_median_5 = sobel_edge(median_5)

# Step (b): 5x5 Median + Gaussian + Sobel
gaussian_after_5 = cv2.GaussianBlur(median_5, (5, 5), 0)
sobel_median5_gaussian = sobel_edge(gaussian_after_5)

# Step (c): 7x7 Median + Gaussian + Sobel
median_7 = cv2.medianBlur(gray, 7)
gaussian_after_7 = cv2.GaussianBlur(median_7, (5, 5), 0)
sobel_median7_gaussian = sobel_edge(gaussian_after_7)

# Display all results
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1), plt.imshow(median_5, cmap='gray'), plt.title('5x5 Median')
plt.subplot(2, 2, 2), plt.imshow(gaussian_after_5, cmap='gray'), plt.title('5x5 Median + Gaussian')
plt.subplot(2, 2, 3), plt.imshow(gaussian_after_7, cmap='gray'), plt.title('7x7 Median + Gaussian')
plt.subplot(2, 2, 4), plt.imshow(gray, cmap='gray'), plt.title('Original Grayscale')
plt.tight_layout()
plt.show()

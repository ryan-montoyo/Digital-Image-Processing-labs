import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and convert to grayscale
image = cv2.imread('IMG_3038.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply median filters
median_5 = cv2.medianBlur(gray, 5)
median_7 = cv2.medianBlur(gray, 7)

# Step 3: Define kernels
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
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(img, dtype=float)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

# Sobel function
def sobel_edge(img):
    gx = manual_convolution(img, sobel_x)
    gy = manual_convolution(img, sobel_y)
    edges = np.sqrt(gx**2 + gy**2)
    return np.clip(edges, 0, 255).astype(np.uint8)

# Step 4: Apply Sobel to filtered images
sobel_5 = sobel_edge(median_5)
sobel_7 = sobel_edge(median_7)

# Step 5: Plot all results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(gray, cmap='gray'), plt.title('Original Grayscale')
plt.subplot(2, 2, 2), plt.imshow(sobel_edge(gray), cmap='gray'), plt.title('Sobel (No Filter)')
plt.subplot(2, 2, 3), plt.imshow(sobel_5, cmap='gray'), plt.title('Sobel after 5x5 Median Filter')
plt.subplot(2, 2, 4), plt.imshow(sobel_7, cmap='gray'), plt.title('Sobel after 7x7 Median Filter')
plt.tight_layout()
plt.show()

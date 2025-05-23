import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and convert to grayscale
img = cv2.imread('IMG_3038.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Convolution function
def manual_convolution(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result

# Step 3: Define kernels
# Roberts (2x2)
roberts_x = np.array([[1, 0], [0, -1]], dtype=float)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=float)

# Prewitt (3x3)
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=float)
prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]], dtype=float)

# Sobel (3x3)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=float)

# Step 4: Apply edge detection with manual convolution
def apply_edge_operator(image, kx, ky):
    gx = manual_convolution(image, kx)
    gy = manual_convolution(image, ky)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

roberts = apply_edge_operator(gray, roberts_x, roberts_y)
prewitt = apply_edge_operator(gray, prewitt_x, prewitt_y)
sobel = apply_edge_operator(gray, sobel_x, sobel_y)

# Step 5: Show results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(gray, cmap='gray'), plt.title('Original')
plt.subplot(2, 2, 2), plt.imshow(roberts, cmap='gray'), plt.title('Roberts')
plt.subplot(2, 2, 3), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
plt.subplot(2, 2, 4), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
plt.tight_layout()
plt.show()

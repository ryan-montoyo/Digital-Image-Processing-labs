import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image and convert to grayscale
image = cv2.imread('snow.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Define kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=float)

# Step 3: Manual convolution function
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

# Step 4: Apply Sobel operator
gx = manual_convolution(gray, sobel_x)
gy = manual_convolution(gray, sobel_y)
sobel_edges = np.sqrt(gx**2 + gy**2)
sobel_edges = np.clip(sobel_edges, 0, 255).astype(np.uint8)

# Step 5: Display result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Snow Image')
plt.subplot(1, 2, 2), plt.imshow(sobel_edges, cmap='gray'), plt.title('Sobel Edge Detection')
plt.tight_layout()
plt.show()

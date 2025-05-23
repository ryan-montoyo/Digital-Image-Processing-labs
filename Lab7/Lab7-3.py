import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os

# Load and prepare the image
img = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)
rows, cols = img.shape

# Ensure dimensions are divisible by 8
rows_pad = rows if rows % 8 == 0 else rows + (8 - rows % 8)
cols_pad = cols if cols % 8 == 0 else cols + (8 - cols % 8)
padded_img = np.zeros((rows_pad, cols_pad))
padded_img[:rows, :cols] = img

# Q matrix
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

# Helper functions
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Compress and reconstruct
compressed = np.zeros_like(padded_img)
for i in range(0, rows_pad, 8):
    for j in range(0, cols_pad, 8):
        block = padded_img[i:i+8, j:j+8]
        dct_block = dct2(block)
        quantized = np.round(dct_block / Q)
        dequantized = quantized * Q
        reconstructed = idct2(dequantized)
        compressed[i:i+8, j:j+8] = reconstructed

# Crop to original size
compressed = compressed[:rows, :cols]

# Display original vs compressed
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(compressed, 0, 255).astype(np.uint8), cmap='gray')
plt.title('Compressed Image')
plt.axis('off')
plt.tight_layout()
plt.show()

# Save to calculate compression ratio
cv2.imwrite('original.bmp', img.astype(np.uint8))       # Uncompressed
cv2.imwrite('compressed.jpg', compressed.astype(np.uint8))  # Compressed

# Compression ratio
original_size = os.path.getsize('original.bmp')
compressed_size = os.path.getsize('compressed.jpg')
compression_ratio = original_size / compressed_size

print(f'Original size: {original_size / 1024:.2f} KB')
print(f'Compressed size: {compressed_size / 1024:.2f} KB')
print(f'Compression Ratio: {compression_ratio:.2f}')

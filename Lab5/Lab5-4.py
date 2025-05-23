import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

# Convert phrase to binary (word-by-word)
phrase = "This is a Golden Retriever"
words = phrase.split()  # ['This', 'is', 'a', 'Golden', 'Retriever']

# Convert each word to binary (8-bit ASCII)
binary_words = [[format(ord(c), '08b') for c in word] for word in words]

# Zero out LSB of entire image before embedding
embedded_img = img.copy()
embedded_img = embedded_img & 0xFE  # Clears the LSBs

for row_idx, word_bin in enumerate(binary_words):
    col = 0
    for char_bin in word_bin:
        for bit in char_bin:
            if col < img.shape[1]:
                pixel = embedded_img[row_idx, col]
                pixel = (pixel & np.uint8(0xFE)) | int(bit)
                embedded_img[row_idx, col] = pixel
                col += 1

# Extract LSBs before and after
lsb_original = img & 1
lsb_embedded = embedded_img & 1

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original LSB")
plt.imshow(lsb_original * 255, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("LSB After Embedding")
plt.imshow(lsb_embedded * 255, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image with Hidden Text")
plt.imshow(embedded_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

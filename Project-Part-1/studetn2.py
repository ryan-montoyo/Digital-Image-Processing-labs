import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing images
img_dir = 'Project-Part-1'
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_channels(image):
    image = image.astype(np.float32) + 1e-6  
    sum_channels = np.sum(image, axis=2, keepdims=True)
    normalized = image / sum_channels
    return normalized

def gamma_correction(image, gamma=0.5):
    corrected = np.power(image, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def alpha_blend(img1, img2):
    alpha_blended = np.clip(img1 * 0.5 + img2 * (1-0.5), 0, 255).astype(np.uint8) # alpha = 0.5
    return alpha_blended

def piecewise_contrast_stretch(img, t):
    img = img.astype(np.float32)
    output = np.zeros_like(img)

    mask1 = img <= t
    mask2 = img > t

    output[mask1] = (img[mask1] / (t + 1e-6)) * 127
    output[mask2] = ((img[mask2] - t) / (255 - t + 1e-6)) * 128 + 127

    return np.clip(output, 0, 255).astype(np.uint8)

def compute_eme(image, window_size=8):
    eme_total = 0
    h, w = image.shape
    for i in range(0, h, window_size):
        for j in range(0, w, window_size):
            block = image[i:i+window_size, j:j+window_size]
            if block.size == 0 or np.min(block) == 0:
                continue
            Imax = np.max(block)
            Imin = np.min(block)
            if Imin == 0:
                Imin = 1  # to avoid division by 0
            eme_total += 20 * np.log10(Imax / Imin)
    return eme_total


for file in img_files:
    # Load image
    path = os.path.join(img_dir, file)
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Convert to grayscale
    gray = rgb_to_gray(img)

    # Step 2: Normalize RGB channels
    norm = normalize_channels(img)

    # Step 3: Difference
    gray_3ch = np.stack([gray]*3, axis=2).astype(np.float32) / 255.0
    difference = np.abs(gray_3ch - norm)

    # Step 4: Gamma correction
    gamma_corrected = gamma_correction(difference)

    # Step 5: Convert gamma-corrected image to grayscale
    gamma_gray = rgb_to_gray(gamma_corrected)

    # Step 6: Alpha blending
    blended = alpha_blend(gray, gamma_gray)

    # Step 7: Contrast enhancement
    best_t = 0
    best_eme = -1
    best_img = None

    for t in range(256):
        stretched = piecewise_contrast_stretch(blended, t)
        eme = compute_eme(stretched)
        if eme > best_eme:
            best_eme = eme
            best_t = t
            best_img = stretched

    print(f"Optimal t = {best_t} with EME = {best_eme}")

    final = best_img

    # Plot all 7 stages
    plt.figure(figsize=(20, 8))
    plt.suptitle(f'Processing: {file}', fontsize=16)

    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(norm)
    plt.title('Normalized RGB')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(difference)
    plt.title('Difference (Norm - Gray)')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.imshow(gamma_corrected, cmap='gray')
    plt.title('Gamma Corrected')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(blended, cmap='gray')
    plt.title('Alpha Blended')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(final, cmap='gray')
    plt.title(f'Contrast Enhanced (t={best_t})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


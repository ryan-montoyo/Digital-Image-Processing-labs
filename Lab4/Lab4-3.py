import cv2
import numpy as np

# Load the image
original = cv2.imread("dog.png", cv2.IMREAD_GRAYSCALE)


# Step 1: Create a low-contrast version
def reduce_contrast(img, low=100, high=150):
    img = img.astype(np.float32)
    img_normalized = (img - img.min()) / (img.max() - img.min())
    compressed = img_normalized * (high - low) + low
    return compressed.astype(np.uint8)

low_contrast = reduce_contrast(original)

def normalize_image(img):
    return ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)

low_contrast_norm = normalize_image(low_contrast)

# Step 2: Enhancement Functions
def histogram_equalization(img):
    return cv2.equalizeHist(img)

def linear_contrast_stretching(img):
    I_min, I_max = np.min(img), np.max(img)
    if I_max - I_min == 0:
        return img.copy()
    stretched = (img - I_min) * 255 / (I_max - I_min)
    return stretched.astype(np.uint8)

def gamma_correction(img, gamma_val=2.2):
    norm = img / 255.0
    corrected = np.power(norm, 1 / gamma_val)
    return (corrected * 255).astype(np.uint8)

def power_law_transform(img, gamma_val=0.5):
    norm = img / 255.0
    transformed = np.power(norm, gamma_val)
    return (transformed * 255).astype(np.uint8)

# Step 3: Apply Enhancements
he_img = histogram_equalization(low_contrast_norm)
lcs_img = linear_contrast_stretching(low_contrast_norm)
gamma_img = gamma_correction(low_contrast_norm, 2.2)
pl_img = power_law_transform(low_contrast_norm, 0.5)


# Step 4: EME Calculation
def calculate_EME(image, M, N):
    image = image.astype(np.float32)
    rows, cols = image.shape
    block_rows = rows // M
    block_cols = cols // N
    eme_total = 0

    for i in range(M):
        for j in range(N):
            block = image[i*block_rows:(i+1)*block_rows, j*block_cols:(j+1)*block_cols]
            I_max = np.max(block)
            I_min = np.min(block)
            if I_min == 0:
                I_min = 1
            if I_max <= I_min:
                continue  # Skip blocks with no variation
            eme_total += 20 * np.log10(I_max / I_min)
    return eme_total / (M * N)


# Step 5: Compute EME values
eme_original = calculate_EME(original, 4, 4)
eme_low = calculate_EME(low_contrast, 4, 4)
eme_he = calculate_EME(he_img, 4, 4)
eme_lcs = calculate_EME(lcs_img, 4, 4)
eme_gamma = calculate_EME(gamma_img, 4, 4)
eme_pl = calculate_EME(pl_img, 4, 4)


# Step 6: Output Results
print(f"Original EME: {eme_original:.2f}")
print(f"Low-Contrast EME: {eme_low:.2f}")
print(f"Histogram Equalization EME: {eme_he:.2f}")
print(f"Linear Contrast Stretching EME: {eme_lcs:.2f}")
print(f"Gamma Correction (γ=2.2) EME: {eme_gamma:.2f}")
print(f"Power-Law Transform (γ=0.5) EME: {eme_pl:.2f}")

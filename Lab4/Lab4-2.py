import cv2
import numpy as np

# Load the grayscale image
img = cv2.imread("dog.png", cv2.IMREAD_GRAYSCALE)

# Function to calculate EME
def calculate_EME(image, M, N):
    image = image.astype(np.float32)
    rows, cols = image.shape
    block_rows = rows // M
    block_cols = cols // N
    eme_total = 0

    for i in range(M):
        for j in range(N):
            # Define block coordinates
            row_start = i * block_rows
            row_end = row_start + block_rows
            col_start = j * block_cols
            col_end = col_start + block_cols

            # Extract block
            block = image[row_start:row_end, col_start:col_end]

            # Compute max and min intensities in the block
            I_max = np.max(block)
            I_min = np.min(block)

            # Avoid division by zero
            if I_min == 0:
                I_min = 1

            # Add EME contribution from the block
            if I_max > 0:
                eme_block = 20 * np.log10(I_max / I_min)
                eme_total += eme_block

    # Normalize by number of blocks
    eme_value = eme_total / (M * N)
    return eme_value

# Apply the function with 8x8 blocks
eme_result = calculate_EME(img, M=8, N=8)
print("EME (Enhancement Measure of Entropy):", round(eme_result, 2))

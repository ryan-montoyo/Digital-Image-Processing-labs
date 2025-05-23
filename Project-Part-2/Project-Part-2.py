import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def otsu_threshold(img):
    # Step 1: Compute histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    # Step 2: Compute probabilities
    total_pixels = img.size
    prob = hist / total_pixels

    # Step 3: Cumulative sum and mean
    cumulative_sum = np.cumsum(prob)
    cumulative_mean = np.cumsum(np.arange(256) * prob)

    # Step 4: Global mean
    global_mean = cumulative_mean[-1]

    # Step 5: Between-class variance
    numerator = (global_mean * cumulative_sum - cumulative_mean)**2
    denominator = cumulative_sum * (1 - cumulative_sum)
    # Avoid division by zero
    valid_mask = denominator != 0
    sigma_b_squared = np.zeros(256)
    sigma_b_squared[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Step 6: Maximize between-class variance
    optimal_threshold = np.argmax(sigma_b_squared)

    # Step 7: Apply threshold
    binary_image = (img >= optimal_threshold).astype(np.uint8) * 255

    return binary_image, optimal_threshold, hist, sigma_b_squared

def process_images():
    image_files = [
        "Project-Part-2/final_CNV-1188386-4_hybrid.jpg",
        "Project-Part-2/final_CNV-1260317-1_hybrid.jpg",
        "Project-Part-2/final_CNV-1290410-2_hybrid.jpg",
        "Project-Part-2/final_DME-30521-11_hybrid.jpg",
        "Project-Part-2/final_DME-1479682-1_hybrid.jpg",
        "Project-Part-2/final_DME-2105194-1_hybrid.jpg",
        "Project-Part-2/final_DRUSEN-1793499-1_hybrid.jpg",
        "Project-Part-2/final_DRUSEN-1997439-1_hybrid.jpg",
        "Project-Part-2/final_DRUSEN-2108193-1_hybrid.jpg",
        "Project-Part-2/final_NORMAL-153950-1_hybrid.jpg"
    ]

    for file in image_files:
        gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        result, t, hist, var = otsu_threshold(gray)

        # Show the result
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Pre-Processed Image")

        plt.subplot(1, 3, 2)
        plt.imshow(result, cmap='gray')
        plt.title(f"Thresholded (T={t})")

        plt.subplot(1, 3, 3)
        plt.plot(var)
        plt.title("Between-Class Variance")
        plt.xlabel("Threshold")
        plt.ylabel("σ²_B")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    process_images()

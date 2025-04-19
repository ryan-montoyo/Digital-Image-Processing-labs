import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images in grayscale mode
I1 = cv2.imread("plane.png", cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread("car.png", cv2.IMREAD_GRAYSCALE)
I3 = cv2.imread("wafer.png", cv2.IMREAD_GRAYSCALE)
I4 = cv2.imread("telescopic.png", cv2.IMREAD_GRAYSCALE)


# Transpose I3
I3T = cv2.transpose(I3)

# Convert to float before operations to prevent overflow/underflow
I1 = I1.astype(np.float32)
I2 = I2.astype(np.float32)
I3T = I3T.astype(np.float32)
I4 = I4.astype(np.float32)

# Perform the arithmetic operation safely
Ioutput = np.clip((I1 + I2) - I3T - (0.3 * I4), 0, 255).astype(np.uint8)

# Display the resulting image
plt.imshow(Ioutput, cmap="gray")
plt.title("(I1+I2) - I3T - (0.3*I4)")
plt.axis("off")
plt.show()

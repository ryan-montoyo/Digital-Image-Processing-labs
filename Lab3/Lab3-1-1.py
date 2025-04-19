import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the 1D sine wave signal
n = np.linspace(0, 4 * np.pi, 500)  # Range from 0 to 4π with 500 points
x = np.sin(n)  # Sine wave signal

# Step 2: Define the averaging and difference filters
h_avg = [1/3, 1/3, 1/3]  # Averaging filter
h_diff = [-1, 1]  # Difference filter

# Convolution function (manual implementation)
def manual_convolution(x, h):
    len_x = len(x)
    len_h = len(h)
    y = np.zeros(len_x)

    # Flip the filter for convolution
    h = h[::-1]

    # Perform convolution
    for i in range(len_x):
        sum = 0
        for j in range(len_h):
            idx = i - (len_h // 2) + j  # Center the kernel
            if 0 <= idx < len_x:  # Only sum valid indices
                sum += x[idx] * h[j]
        y[i] = sum

    return y

# Step 3: Apply manual convolution with both filters
y_avg = manual_convolution(x, h_avg)
y_diff = manual_convolution(x, h_diff)

# Step 4: Plot the original signal and the filtered signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(n, x, label='Original Signal', color='blue')
plt.title('Original Signal (sin(2πn))')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(n, y_avg, label='Averaged Signal', color='green')
plt.title('Averaged Signal (Manual Convolution with Averaging Filter)')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, y_diff, label='Difference Signal', color='red')
plt.title('Difference Signal (Manual Convolution with Difference Filter)')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

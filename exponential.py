import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Define the rate parameter for the exponential distribution
rate = 0.9  

# Generate a range of values for the x-axis
x = np.linspace(0, 10, 100)

# Calculate the PDF 
pdf = expon.pdf(x, scale=1/rate)

# Calculate the CDF 
cdf = expon.cdf(x, scale=1/rate)

# Calculate the mean and variance
mean = 1 / rate
variance = 1 / (rate ** 2)

# Plot the PDF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'r-', lw=2)
plt.title('Exponential PDF')
plt.xlabel('X')
plt.ylabel('Probability Density')

# Plot the CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'b-', lw=2)
plt.title('Exponential CDF')
plt.xlabel('X')
plt.ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()

# Print statistics
print("\n Statistics:")
print(f"Mean: {mean:.5f}")
print(f"Variance: {variance:.5f}")

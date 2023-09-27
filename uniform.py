import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Define the rate parameter 
rate = 0.9  

# Define the number of random variables to generate
num_samples = 10000

# Generate random variables from a uniform distribution (0, 1)
uniform_samples = np.random.rand(num_samples)

# Transform uniform samples to exponential distribution using the inverse transform method
exponential_samples = -np.log(1 - uniform_samples) / rate

# Calculate and display statistics of the generated exponential samples
mean = np.mean(exponential_samples)
variance = np.var(exponential_samples)
std_deviation = np.sqrt(variance)
percentiles = np.percentile(exponential_samples, [25, 50, 75])

print(f"Mean: {mean:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Standard Deviation: {std_deviation:.4f}")
print(f"25th Percentile: {percentiles[0]:.4f}")
print(f"50th Percentile (Median): {percentiles[1]:.4f}")
print(f"75th Percentile: {percentiles[2]:.4f}")

# Plot a histogram of the generated exponential samples
plt.figure(figsize=(12, 8))

# Histogram
plt.subplot(2, 2, 1)
plt.hist(exponential_samples, bins=30, density=True, alpha=0.6, color='blue', label='Histogram')
plt.title('Exponential Distribution (PDF)')
plt.xlabel('X')
plt.ylabel('Probability Density')

# PDF
x = np.linspace(0, 10, 100)
pdf = expon.pdf(x, scale=1/rate)
plt.subplot(2, 2, 2)
plt.plot(x, pdf, 'r-', lw=2, label='PDF')
plt.title('PDF')
plt.xlabel('X')
plt.ylabel('Probability Density')

# CDF
cdf = expon.cdf(x, scale=1/rate)
plt.subplot(2, 2, 3)
plt.plot(x, cdf, 'g-', lw=2, label='CDF')
plt.title('CDF')
plt.xlabel('X')
plt.ylabel('Cumulative Probability')

# Percentiles
plt.subplot(2, 2, 4)
plt.boxplot(exponential_samples, vert=False)
plt.title('Percentiles (Boxplot)')
plt.xlabel('X')

plt.tight_layout()
plt.show()

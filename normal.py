import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters 
mean_value = 5.0  # Mean of the normal distribution
std_deviation = 2.0  # Standard deviation of the normal distribution

# Generate a range of values for the x-axis
x = np.linspace(-2, 12, 400)

# Calculate the PDF
pdf = norm.pdf(x, loc=mean_value, scale=std_deviation)

# Calculate the CDF 
cdf = norm.cdf(x, loc=mean_value, scale=std_deviation)

# Calculate the mean and variance
mean = norm.mean(loc=mean_value, scale=std_deviation)
variance = norm.var(loc=mean_value, scale=std_deviation)

# Computations:
# Probability of observing a value less than 7
prob_less_than_7 = norm.cdf(7, loc=mean_value, scale=std_deviation)

# Probability of observing a value greater than 3
prob_greater_than_3 = 1 - norm.cdf(3, loc=mean_value, scale=std_deviation)

# Probability of observing a value between 4 and 6
prob_between_4_and_6 = norm.cdf(6, loc=mean_value, scale=std_deviation) - norm.cdf(4, loc=mean_value, scale=std_deviation)

# Print the PDF and CDF
print("Normal PDF:")
for val, prob in zip(x, pdf):
    print(f"X = {val:.2f}: PDF = {prob:.5f}")

print("\nNormal CDF:")
for val, prob in zip(x, cdf):
    print(f"X = {val:.2f}: CDF = {prob:.5f}")

# Print mean and variance
print(f"\nMean of the Normal distribution: {mean:.5f}")
print(f"Variance of the Normal distribution: {variance:.5f}")

# Print probabilities
print(f"\nProbability of observing a value less than 7: {prob_less_than_7:.5f}")
print(f"Probability of observing a value greater than 3: {prob_greater_than_3:.5f}")
print(f"Probability of observing a value between 4 and 6: {prob_between_4_and_6:.5f}")

# Plot Normal PDF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'r-', lw=2)
plt.title('Normal PDF')
plt.xlabel('X')
plt.ylabel('Probability Density')

# Plot Normal CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'b-', lw=2)
plt.title('Normal CDF')
plt.xlabel('X')
plt.ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()

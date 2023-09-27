import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define parameters 
lambda_value = 9  # Mean rate (λ) 
k_values = np.arange(0, 20)  # Range of k values to calculate

# Calculate the Poisson PMF 
pmf_values = poisson.pmf(k_values, mu=lambda_value)

# Calculate the Poisson CDF 
cdf_values = poisson.cdf(k_values, mu=lambda_value)

# Calculate the mean and variance
mean_value = poisson.mean(mu=lambda_value)
variance_value = poisson.var(mu=lambda_value)

# Computations:
# Probability of observing more than 10 events
prob_more_than_10 = 1 - poisson.cdf(10, mu=lambda_value)

# Probability of observing fewer than 5 events
prob_less_than_5 = poisson.cdf(4, mu=lambda_value)

# Probability of observing between 5 and 10 events
prob_between_5_and_10 = poisson.cdf(10, mu=lambda_value) - poisson.cdf(4, mu=lambda_value)

# Print the Poisson PMF and CDF
print("Poisson PMF:")
for k, prob in zip(k_values, pmf_values):
    print(f"k = {k}: P(k) = {prob:.5f}")

print("\nPoisson CDF:")
for k, prob in zip(k_values, cdf_values):
    print(f"k = {k}: F(k) = {prob:.5f}")

# Print mean and variance
print(f"\nMean of the Poisson distribution: {mean_value:.5f}")
print(f"Variance of the Poisson distribution: {variance_value:.5f}")

# Print probabilities
print(f"\nProbability of observing more than 10 events: {prob_more_than_10:.5f}")
print(f"Probability of observing fewer than 5 events: {prob_less_than_5:.5f}")
print(f"Probability of observing between 5 and 10 events: {prob_between_5_and_10:.5f}")

# Plot Poisson PMF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(k_values, pmf_values, align='center', alpha=0.7)
plt.xlabel('Number of Events (k)')
plt.ylabel('Probability')
plt.title('Poisson PMF for λ = 9')

# Plot Poisson CDF
plt.subplot(1, 2, 2)
plt.step(k_values, cdf_values, where='mid', label='CDF', color='orange')
plt.xlabel('Number of Events (k)')
plt.ylabel('Cumulative Probability')
plt.title('Poisson CDF for λ = 9')

plt.tight_layout()
plt.show()

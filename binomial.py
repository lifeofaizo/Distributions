import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, binom_test

# Define parameters 
num_trials = 20  # Number of trials 
probability_success = 0.4  # Probability of success in each trial

# Generate a range of k values (number of successes)
num_successes = np.arange(0, num_trials + 1)

# Calculate the Binomial PMF
binomial_pmf = binom.pmf(num_successes, num_trials, probability_success)

# Calculate the Binomial CDF 
binomial_cdf = binom.cdf(num_successes, num_trials, probability_success)

# Print the Binomial PMF and CDF
print("Binomial PMF:")
for k, prob in zip(num_successes, binomial_pmf):
    print(f"Number of Successes (k) = {k}: PMF = {prob:.5f}")

print("\nBinomial CDF:")
for k, prob in zip(num_successes, binomial_cdf):
    print(f"Number of Successes (k) = {k}: CDF = {prob:.5f}")

# Computations
mean = num_trials * probability_success  # Mean
variance = num_trials * probability_success * (1 - probability_success)  # Variance
std_deviation = np.sqrt(variance)  # Standard Deviation

# Print Computations
print("\nAdditional Statistics:")
print(f"Mean: {mean:.5f}")
print(f"Variance: {variance:.5f}")
print(f"Standard Deviation: {std_deviation:.5f}")

# Plot Binomial PMF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(num_successes, binomial_pmf, align='center', alpha=0.7)
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability')
plt.title(f'Binomial PMF for {num_trials} Trials, p = {probability_success}')

# Plot Binomial CDF
plt.subplot(1, 2, 2)
plt.step(num_successes, binomial_cdf, where='mid', label='CDF', color='orange')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Cumulative Probability')
plt.title(f'Binomial CDF for {num_trials} Trials, p = {probability_success}')

plt.tight_layout()
plt.show()

# Perform a Binomial Test
observed_successes = 8  # Observed number of successes
alpha = 0.05  # Significance level

# Perform the binomial test
p_value = binom_test(observed_successes, num_trials, probability_success, alternative='greater')

# Print the result of the binomial test
print(f"\nBinomial Test Result:")
print(f"Observed Successes: {observed_successes}")
print(f"p-value: {p_value}")

# Determine the outcome of the test based on the p-value and significance level
if p_value < alpha:
    print("Reject the null hypothesis: The die is biased towards success.")
else:
    print("Fail to reject the null hypothesis: There is no significant bias towards success.")
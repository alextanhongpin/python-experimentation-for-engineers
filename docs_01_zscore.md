# Z-score

- https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
- https://open.maricopa.edu/psy230mm/chapter/chapter-1-z-test-hypothesis-testing/
- https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/hypothesis-testing/one-tailed-and-two-tailed-tests.html

To calculate a z-score, you can use the formula:

$\[ z = \frac{(X - \mu)}{\sigma} \]$

where:
- $\( X \)$ is the value you want to standardize.
- $\( \mu \)$ is the mean of the dataset.
- $\( \sigma \)$ is the standard deviation of the dataset.

Here's an example in Python:

```python
import numpy as np

# Example data
data = np.array([10, 20, 30, 40, 50])

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Calculate z-scores
z_scores = (data - mean) / std_dev

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Z-scores:", z_scores)
```

This code calculates the mean and standard deviation of the `data` array, then computes the z-scores for each element in the array.

## Comparing two distributions using p-value

To use z-scores to compare two distributions and test a null hypothesis, you typically follow these steps:

1. **Calculate the means and standard deviations** of the two distributions.
2. **Compute the z-score** for the difference between the two means.
3. **Determine the p-value** from the z-score to test the null hypothesis.

Here's a step-by-step example in Python:

```python
import numpy as np
from scipy import stats

# Example data for two distributions
data1 = np.array([10, 20, 30, 40, 50])
data2 = np.array([15, 25, 35, 45, 55])

# Calculate means and standard deviations
mean1, mean2 = np.mean(data1), np.mean(data2)
std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

# Calculate the z-score for the difference between the two means
n1, n2 = len(data1), len(data2)
pooled_std = np.sqrt((std1**2 / n1) + (std2**2 / n2))
z_score = (mean1 - mean2) / pooled_std

# Determine the p-value from the z-score
p_value = stats.norm.sf(abs(z_score)) * 2  # two-tailed test

# Compare with a significance level (alpha)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

print(f"Mean1: {mean1}, Mean2: {mean2}")
print(f"Z-score: {z_score}")
print(f"P-value: {p_value}")
```

In this example:
- `data1` and `data2` represent the two distributions.
- We calculate the means and standard deviations for each distribution.
- We then compute the z-score for the difference between the two means.
- Using the z-score, we find the p-value to test the null hypothesis (e.g., whether the means are significantly different).

If the p-value is less than the significance level (e.g., 0.05), we reject the null hypothesis, indicating a significant difference between the distributions. Otherwise, we fail to reject the null hypothesis.

## Using zscore instead to reject null hypothesis

To use a z-score instead of a p-value for testing a null hypothesis, you can follow these steps:

1. **Calculate the z-score**: Use the formula to compute the z-score for the difference between two sample means.
2. **Determine the critical value**: Identify the critical z-value for the desired significance level (e.g., 1.96 for a two-tailed test at the 0.05 significance level).
3. **Compare the z-score to the critical value**: Reject the null hypothesis if the absolute value of the z-score is greater than the critical value.

Here's an example in Python:

```python
import numpy as np

# Example data for two distributions
data1 = np.array([10, 20, 30, 40, 50])
data2 = np.array([15, 25, 35, 45, 55])

# Calculate means and standard deviations
mean1, mean2 = np.mean(data1), np.mean(data2)
std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

# Calculate the z-score for the difference between the two means
n1, n2 = len(data1), len(data2)
pooled_std = np.sqrt((std1**2 / n1) + (std2**2 / n2))
z_score = (mean1 - mean2) / pooled_std

# Determine the critical value for a two-tailed test at the 0.05 significance level
alpha = 0.05
critical_value = 1.96  # for a two-tailed test

# Compare the z-score to the critical value
if abs(z_score) > critical_value:
    result = "Reject the null hypothesis"
else:
    result = "Fail to reject the null hypothesis"

print(f"Mean1: {mean1}, Mean2: {mean2}")
print(f"Z-score: {z_score}")
print(f"Critical value: {critical_value}")
print(f"Result: {result}")
```

In this example:
- `data1` and `data2` are the two distributions.
- We calculate the means and standard deviations for each distribution.
- The z-score for the difference between the two means is computed.
- We compare the absolute value of the z-score to the critical value (1.96 for a two-tailed test at the 0.05 significance level).
- If the absolute value of the z-score is greater than the critical value, we reject the null hypothesis, indicating a significant difference between the distributions. Otherwise, we fail to reject the null hypothesis.

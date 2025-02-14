# A/B testing


Sample size calculation:
- https://mverbakel.github.io/2021-04-11/power-calculations
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5738522/
- https://www.reddit.com/r/AskStatistics/comments/1i2ran5/standard_deviation_in_sample_size_calculation_for/
- https://ethanweed.github.io/pythonbook/05.02-ttest.html


```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import statsmodels.stats.api as sm
import statsmodels.stats.power as sp
```


```python
import math

import scipy.stats as stats


def sample_size_for_mean_difference(alpha, power, sigma, delta):
    """
    Calculate sample size for comparing two means.

    Parameters:
        alpha (float): significance level
        power (float): statistical power
        sigma (float): standard deviation of the differences
        delta (float): the difference in means to detect

    Returns:
        float: required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) * sigma / delta) ** 2
    return math.ceil(n)


def sample_size_for_proportions(alpha, power, p1, p2):
    """
    Calculate sample size for comparing two proportions.

    Parameters:
        alpha (float): significance level
        power (float): statistical power
        p1 (float): proportion in group 1
        p2 (float): proportion in group 2

    Returns:
        float: required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_mean = (p1 + p2) / 2
    n = ((z_alpha + z_beta) ** 2 * (p_mean * (1 - p_mean) * 2)) / (p1 - p2) ** 2
    return math.ceil(n)


# Example calculations
alpha = 0.05  # 95% confidence level
power = 0.80  # 80% power

# Mean difference example
sigma = 10
delta = 5
n_mean_diff = sample_size_for_mean_difference(alpha, power, sigma, delta)
print(f"Required sample size per group for mean difference: {n_mean_diff}")

# Proportion example
p1 = 0.5
p2 = 0.4
n_proportion = sample_size_for_proportions(alpha, power, p1, p2)
print(f"Required sample size per group for proportions: {n_proportion}")
```

    Required sample size per group for mean difference: 32
    Required sample size per group for proportions: 389



```python
def sample_size_for_mean_difference(delta, std, power=0.8, alpha=0.05, sides=1):
    return math.ceil(
        sp.normal_sample_size_one_tail(
            delta, power, alpha / sides, std_null=std, std_alternative=None
        )
    )
```


```python
one_tail = sample_size_for_mean_difference(delta, sigma, sides=1)
two_tail = sample_size_for_mean_difference(delta, sigma, sides=2)
one_tail, two_tail
```




    (25, 32)




```python
def sample_size_for_proportions(p1, p2, power=0.8, alternative="two-sided"):
    return math.ceil(
        sm.samplesize_proportions_2indep_onetail(
            p2 - p1, p1, power, alternative=alternative
        )
    )


one_tail = sample_size_for_proportions(p1, p2, alternative="larger")
two_tail = sample_size_for_proportions(p1, p2, alternative="two-sided")
one_tail, two_tail
```




    (305, 388)




```python
n = 32
a = np.random.normal(1, sigma, n)
b = np.random.normal(1 + delta, sigma, n)
sm.ztest(a, b)
```




    (-1.2011498999324322, 0.2296930594566906)




```python
sm.ttest_ind(a, b)
```




    (-1.2011498999324322, 0.23426165878620456, 62.0)




```python
nobs1 = 388
nobs2 = 388
sm.test_proportions_2indep(
    p1 * nobs1, nobs1, p2 * nobs2, nobs2, return_results=False  # count1,
)
```




    (2.806441871268021, 0.0050091949361540785)




```python
count = [p1 * nobs1, p2 * nobs2]
nobs = [nobs1, nobs2]
sm.proportions_ztest(count, nobs, alternative="two-sided", prop_var=False)
```




    (2.799711384836629, 0.005114831546164752)



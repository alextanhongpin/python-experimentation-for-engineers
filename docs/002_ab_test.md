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
sm.power_proportions_2indep(
    p1 - p2,
    p2,
    nobs1=1455,
    ratio=1,
    alpha=0.05,
    alternative="two-sided",
    return_results=True,
)
```




    <class 'statsmodels.tools.testing.Holder'>
    power = 0.9997486095693934
    p_pooled = 0.45
    std_null = 0.7035623639735145
    std_alt = 0.7
    nobs1 = 1455
    nobs2 = 1455
    nobs_ratio = 1
    alpha = 0.05




```python
n = 33
np.random.seed(4)
a = np.random.normal(0, sigma, n)
b = np.random.normal(delta, sigma, n)
sm.ztest(a, b)
```




    (-2.026170300843182, 0.04274733595219376)




```python
np.mean(a), np.std(a), np.mean(b), np.std(b)
```




    (0.30494685217315737, 9.361662707497304, 5.286954602816181, 10.28718941558651)




```python
sm.ttest_ind(a, b)
```




    (-2.026170300843183, 0.046919745963708216, 64.0)




```python
st.ttest_ind(a, b)
```




    TtestResult(statistic=-2.026170300843182, pvalue=0.04691974596370832, df=64.0)




```python
st.ttest_ind_from_stats(
    np.mean(a),
    np.std(a, ddof=1),
    len(a),
    np.mean(b),
    np.std(b, ddof=1),
    len(b),
)
```




    Ttest_indResult(statistic=-2.026170300843182, pvalue=0.04691974596370832)




```python
# nobs1 = 388
# nobs2 = 388
# count1 = math.ceil(p1 * nobs1)
# count2 = math.ceil(p2 * nobs2)
# count1 = 1600
# nobs1 = 80000
# count2 = 1696
# nobs2 = 80000

count1 = 50
nobs1 = 500
count2 = 100
nobs2 = 600
p1 = count1 / nobs1
p2 = count2 / nobs2

sm.test_proportions_2indep(
    count1,
    nobs1,
    count2,
    nobs2,
    return_results=True,
)
```




    <class 'statsmodels.stats.base.HolderTuple'>
    statistic = -3.2535580685929917
    pvalue = 0.0011396942939908236
    compare = 'diff'
    method = 'agresti-caffo'
    diff = -0.06666666666666665
    ratio = 0.6000000000000001
    odds_ratio = 0.5555555555555557
    variance = 0.00041375423296553056
    alternative = 'two-sided'
    value = 0
    tuple = (-3.2535580685929917, 0.0011396942939908236)




```python
import statsmodels.stats.proportion as sap

sap.score_test_proportions_2indep(
    count1,
    nobs1,
    count2,
    nobs2,
    return_results=False,
)
```




    (-3.206718094200441, 0.001342584922266084)




```python
count = [count1, count2]
nobs = [nobs1, nobs2]
sm.proportions_ztest(count, nobs, alternative="two-sided", prop_var=False)
```




    (-3.2081766879052513, 0.0013357940586552824)




```python
n1 = nobs1
n2 = nobs2
# p = (p1 * n1 + p2 * n2) / (n1 + n2)
p = (count1 + count2) / (n1 + n2)
std = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
print(p)
print(std)

# This has the closest result to abtest
# https://abtestguide.com/calc/
st.ttest_ind_from_stats(
    p1,
    np.sqrt(p1 * (1 - p1)),
    nobs1,
    p2,
    np.sqrt(p2 * (1 - p2)),
    nobs2,
    equal_var=False,
)
```

    0.13636363636363635
    0.02078023536484084





    Ttest_indResult(statistic=-3.286499674123032, pvalue=0.0010465503504400564)




```python
std_a = np.sqrt((p1 * (1 - p1)) / nobs1)
std_b = np.sqrt((p2 * (1 - p2)) / nobs2)
std_diff = np.sqrt(std_a**2 + std_b**2)
z_score = (p1 - p2) / (std_diff)
p_value = st.norm.sf(np.abs(z_score))  # One-sided. To get the two-sided, multiply by 2.
z_score, p_value
```




    (-3.286499674123032, 0.000507204456283015)




```python
sm.proportion_effectsize(p2, p1)
```




    0.19756756177464585




```python
sm.proportion_confint(count1, nobs1, alpha=0.05, method="normal")
```




    (0.07370432378270256, 0.12629567621729745)




```python
sm.power_proportions_2indep(
    p1 - p2,
    p2,
    nobs1,
    ratio=nobs2 / nobs1,
    alpha=0.05,
    alternative="two-sided",
    return_results=True,
)
```




    <class 'statsmodels.tools.testing.Holder'>
    power = 0.8994962353514785
    p_pooled = 0.13636363636363635
    std_null = 0.4646601886422926
    std_alt = 0.4535865305988933
    nobs1 = 500
    nobs2 = 600.0
    nobs_ratio = 1.2
    alpha = 0.05




```python
p1 - p2, p2 + (p1 - p2) == p1
```




    (-0.06666666666666665, True)




```python
sm.zt_ind_solve_power(
    sm.proportion_effectsize(p1, p2),
    nobs1,
    alpha=0.05,
    power=None,
    ratio=nobs2 / nobs1,
    alternative="two-sided",
)
```




    0.9036712061263761




```python
sm.tt_ind_solve_power(
    sm.proportion_effectsize(p2, p1),
    nobs1,
    alpha=0.05,
    power=None,
    ratio=nobs2 / nobs1,
)
```




    0.9031827828884099




```python
import statsmodels.stats.api as sms


def calculate_ab_test_power(p1, p2, n1, n2, alpha=0.05):
    # Calculate effect size
    effect_size = sms.proportion_effectsize(p1, p2)

    # Calculate power
    power_analysis = sms.NormalIndPower()
    power = power_analysis.solve_power(
        effect_size=effect_size,
        nobs1=n1,
        alpha=alpha,
        ratio=n2 / n1,
        alternative="two-sided",
    )

    return power


# Example usage

power = calculate_ab_test_power(p1, p2, n1, n2, alpha)
print(f"Calculated power for the A/B test: {power:.2f}")
```

    Calculated power for the A/B test: 0.90


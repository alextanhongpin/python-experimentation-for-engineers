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
    nobs1=388,
    ratio=1,
    alpha=0.05,
    alternative="two-sided",
    return_results=True,
)
```




    <class 'statsmodels.tools.testing.Holder'>
    power = 0.7809360655972347
    p_pooled = 0.13333333333333333
    std_null = 0.48074017006186526
    std_alt = 0.47842333648024415
    nobs1 = 388
    nobs2 = 388
    nobs_ratio = 1
    alpha = 0.05




```python
n = 33
np.random.seed(4)
a = np.random.normal(0, sigma, n)
b = np.random.normal(delta, sigma, n)
sm.ztest(a, b)
```




    (-0.019795765823731406, 0.9842062956060404)




```python
np.mean(a), np.std(a), np.mean(b), np.std(b)
```




    (0.30494685217315737,
     9.361662707497304,
     0.35362126948284883,
     10.287189415586512)




```python
sm.ttest_ind(a, b)
```




    (-0.019795765823731385, 0.9842678829762332, 64.0)




```python
st.ttest_ind(a, b)
```




    TtestResult(statistic=-0.019795765823731406, pvalue=0.9842678829762332, df=64.0)




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




    Ttest_indResult(statistic=-0.019795765823731406, pvalue=0.9842678829762332)




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
count2 = 90
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
    statistic = -2.494217538993216
    pvalue = 0.012623515252023227
    compare = 'diff'
    method = 'agresti-caffo'
    diff = -0.04999999999999999
    ratio = 0.6666666666666667
    odds_ratio = 0.6296296296296297
    variance = 0.00039496130786195586
    alternative = 'two-sided'
    value = 0
    tuple = (-2.494217538993216, 0.012623515252023227)




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




    (-2.476451594519868, 0.01326956172775838)




```python
count = [count1, count2]
nobs = [nobs1, nobs2]
sm.proportions_ztest(count, nobs, alternative="two-sided", prop_var=False)
```




    (-2.4775780224127866, 0.013227748427113117)




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

    0.12727272727272726
    0.02018099916438052





    Ttest_indResult(statistic=-2.5237723256253433, pvalue=0.011750495881847919)




```python
std_a = np.sqrt((p1 * (1 - p1)) / nobs1)
std_b = np.sqrt((p2 * (1 - p2)) / nobs2)
std_diff = np.sqrt(std_a**2 + std_b**2)
z_score = (p1 - p2) / (std_diff)
p_value = st.norm.sf(np.abs(z_score))  # One-sided. To get the two-sided, multiply by 2.
z_score, p_value
```




    (-2.5237723256253433, 0.005805154770786605)




```python
sm.proportion_effectsize(p2, p1)
```




    0.15189772139085922




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
    power = 0.7009982730689444
    p_pooled = 0.1272727272727273
    std_null = 0.451260859854213
    std_alt = 0.4430011286667338
    nobs1 = 500
    nobs2 = 600.0
    nobs_ratio = 1.2
    alpha = 0.05




```python
p1 - p2, p2 + (p1 - p2) == p1
```




    (-0.04999999999999999, True)




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




    0.7083448374710821




```python
sm.tt_ind_solve_power(
    sm.proportion_effectsize(p2, p1),
    nobs1,
    alpha=0.05,
    power=None,
    ratio=nobs2 / nobs1,
)
```




    0.7075911983838853




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


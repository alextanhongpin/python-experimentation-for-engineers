# 


```python
import math

import numpy as np
import scipy.stats as st
import statsmodels.stats.api as sm
import statsmodels.stats.power as sp
```


```python
power = 0.8
alpha = 0.05
ratio = 1

p1 = 0.02
p2 = p1 * 0.15 + p1
p1 = 0.01
p2 = p1 * 0.14 + p1
p1, p2
```




    (0.01, 0.0114)




```python
nobs1 = math.ceil(
    sm.samplesize_proportions_2indep_onetail(
        p2 - p1, p1, power, alternative="two-sided", ratio=ratio
    )
)
nobs1
```




    84779




```python
nobs2 = nobs1
count1 = math.ceil(p1 * nobs1)
count2 = math.ceil(p2 * nobs2)
count1, count2, count1 / nobs1, count2 / nobs2
```




    (848, 967, 0.010002477028509418, 0.011406126517179963)




```python
sm.test_proportions_2indep(
    count1,
    nobs1,
    count2,
    nobs2,
    method="wald",  # Must be 'wald'
    return_results=True,
)
```




    <class 'statsmodels.stats.base.HolderTuple'>
    statistic = -2.808378805210456
    pvalue = 0.004979161993152016
    compare = 'diff'
    method = 'wald'
    diff = -0.0014036494886705449
    ratio = 0.8769389865563597
    odds_ratio = 0.8756956350009129
    variance = 2.498077858533081e-07
    alternative = 'two-sided'
    value = 0
    tuple = (-2.808378805210456, 0.004979161993152016)




```python
# This is closer to abtestguide
sm.proportions_ztest(
    count=[count1, count2], nobs=[nobs1, nobs2], alternative="two-sided", prop_var=False
)
```




    (-2.808313491609461, 0.004980172048271678)




```python
sm.power_proportions_2indep(
    p1 - p2,
    p2,
    nobs1,
    ratio=nobs2 / nobs1,
    alpha=alpha,
    alternative="two-sided",
    return_results=True,
)
```




    <class 'statsmodels.tools.testing.Holder'>
    power = 0.800001811889412
    p_pooled = 0.010700000000000001
    std_null = 0.14550264602405003
    std_alt = 0.14549927834872584
    nobs1 = 84779
    nobs2 = 84779.0
    nobs_ratio = 1.0
    alpha = 0.05




```python
def evan_miller_sample_size(p, delta, alpha=0.05, power=0.8):
    if p > 0.5:
        p = 1.0 - p
    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_beta = st.norm.ppf(power)

    sd1 = np.sqrt(2 * p * (1 - p))
    sd2 = np.sqrt(p * (1 - p) + (p + delta) * (1 - p - delta))
    return math.ceil(
        (
            (z_alpha * sd1 + z_beta * sd2)
            * (z_alpha * sd1 + z_beta * sd2)
            / (delta * delta)
        )
    )


evan_miller_sample_size(p1, p2 - p1)
```




    80919



# Comparing Two Independent Population Proportions

$$
z = \frac{(p_1 - p_2) - D_o}{\sqrt{
\frac{p_1\cdot(1-p_1)}{n_1} + \frac{p_2\cdot(1-p_2)}{n_2}}} $$


```python
alpha = 0.10
delta = 0.05

p1 = 0.67
p2 = 0.8
n1 = 500
n2 = 100
value = -0.05


s1 = p1 * (1 - p1)
s2 = p2 * (1 - p2)
se = np.sqrt(s1 / n1 + s2 / n2)
z = ((p1 - p2) - value) / se
p = st.norm.sf(np.abs(z))
z, p, p * 2
```




    (-1.7702754679970862, 0.03834063134876544, 0.07668126269753088)




```python
count1 = math.ceil(p1 * n1)
count2 = math.ceil(p2 * n2)
nobs1 = n1
nobs2 = n2
ratio = nobs2 / nobs1
sm.test_proportions_2indep(
    count1,
    nobs1,
    count2,
    nobs2,
    return_results=True,
    value=value,
    method="wald",
    alternative="two-sided",
)
```




    <class 'statsmodels.stats.base.HolderTuple'>
    statistic = -1.7702754679970862
    pvalue = 0.07668126269753088
    compare = 'diff'
    method = 'wald'
    diff = -0.13
    ratio = 0.8375
    odds_ratio = 0.5075757575757575
    variance = 0.0020421999999999997
    alternative = 'two-sided'
    value = -0.05
    tuple = (-1.7702754679970862, 0.07668126269753088)




```python
sm.proportions_ztest(
    count=[count1, count2],
    nobs=[nobs1, nobs2],
    alternative="two-sided",
    prop_var=False,
    value=value,
)
```




    (-1.5813962397448424, 0.11378746671001491)



# Sample size calculation for comparing proportions


$n = (Z_\text{α/2}+Z_β)^2 * (p_1(1-p_1)+p_2(1-p_2)) / (p_1-p_2)^2$[^1]


[^1]: https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/



```python
p1 = 0.2
p2 = 0.3

ratio = 1
alpha = 0.05

s1 = p1 * (1 - p1)
s2 = p2 * (1 - p2)

z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
n = (z**2 * (s1 + s2)) / (p1 - p2) ** 2
n
```




    290.4085501709164




```python
n1 = n2 = math.ceil(n)
value = 0
s1 = p1 * (1 - p1)
s2 = p2 * (1 - p2)
se = np.sqrt(s1 / n1 + s2 / n2)
z = ((p1 - p2) - value) / se
p = st.norm.sf(np.abs(z))
z, p, p * 2
```




    (-2.804436639481245, 0.0025202296674208865, 0.005040459334841773)




```python
# This is more accurate than the one above
# Stick to statsmodels for both the calculation.
nobs1 = math.ceil(
    sm.samplesize_proportions_2indep_onetail(
        p1 - p2, p2, power=power, alpha=alpha, alternative="two-sided", ratio=ratio
    )
)
nobs1
```




    294




```python
nobs2 = nobs1
count1 = math.ceil(p1 * nobs1)
count2 = math.ceil(p2 * nobs2)
sm.test_proportions_2indep(
    count1,
    nobs1,
    count2,
    nobs2,
    return_results=True,
    value=0,
    method="wald",
    alternative="two-sided",
)
```




    <class 'statsmodels.stats.base.HolderTuple'>
    statistic = -2.87061379221714
    pvalue = 0.0040967571509745926
    compare = 'diff'
    method = 'wald'
    diff = -0.10204081632653061
    ratio = 0.6629213483146067
    odds_ratio = 0.5782930910829548
    variance = 0.0012635671141055802
    alternative = 'two-sided'
    value = 0
    tuple = (-2.87061379221714, 0.0040967571509745926)



## Difference between method `wald` and `agresti-caffo`:

https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/diffprop.htm


For `wald` (aka _normal approximation_):

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2}
                 \sqrt{\frac{\hat{p}_1 (1 - \hat{p}_1)}{n_1} +
                 \frac{\hat{p}_2 (1 - \hat{p}_2)}{n_2}}
$$

For `agresti-caffo` (aka _adjusted wald_):

$$
(\tilde{p}_1 - \tilde{p}_2) \pm z_{\alpha/2}
                 \sqrt{\frac{\tilde{p}_1 (1 - \tilde{p}_1)}{n_1+2} +
                 \frac{\tilde{p}_2 (1 - \tilde{p}_2)}{n_2+2}}
$$

`wald` method is commonly used. However, Agresti and Caffo pointed out that this method does not always perform well in the sense that the actual coverage probabilities can be less than (and often substantially less than) the nominal coverage probabilities.




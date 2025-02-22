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
    return_results=True,
)
```




    <class 'statsmodels.stats.base.HolderTuple'>
    statistic = -2.8068490684958927
    pvalue = 0.005002867614953665
    compare = 'diff'
    method = 'agresti-caffo'
    diff = -0.0014036494886705449
    ratio = 0.8769389865563597
    odds_ratio = 0.8756956350009129
    variance = 2.500683525084858e-07
    alternative = 'two-sided'
    value = 0
    tuple = (-2.8068490684958927, 0.005002867614953665)




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



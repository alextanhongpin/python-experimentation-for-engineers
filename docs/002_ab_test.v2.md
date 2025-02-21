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
p2 = p1 * 1.15
p1, p2
```




    (0.02, 0.023)




```python
nobs1 = math.ceil(
    sm.samplesize_proportions_2indep_onetail(
        p2 - p1, p1, power, alternative="two-sided", ratio=ratio
    )
)
nobs1
```




    36693




```python
nobs2 = nobs1
count1 = math.ceil(p1 * nobs1)
count2 = math.ceil(p2 * nobs2)
count1, count2, count1 / nobs1, count2 / nobs2
```




    (734, 844, 0.02000381544163737, 0.02300166244242771)




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
    statistic = -2.7977781958042582
    pvalue = 0.005145543433134009
    compare = 'diff'
    method = 'agresti-caffo'
    diff = -0.0029978470007903414
    ratio = 0.8696682464454977
    odds_ratio = 0.8670078969611125
    variance = 1.1480082889411636e-06
    alternative = 'two-sided'
    value = 0
    tuple = (-2.7977781958042582, 0.005145543433134009)




```python
# This is closer to abtestguide
sm.proportions_ztest(
    count=[count1, count2], nobs=[nobs1, nobs2], alternative="two-sided", prop_var=False
)
```




    (-2.7993640341603294, 0.0051203375431128784)




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
    power = 0.8000034194420894
    p_pooled = 0.0215
    std_null = 0.20512313375141283
    std_alt = 0.2051121644369246
    nobs1 = 36693
    nobs2 = 36693.0
    nobs_ratio = 1.0
    alpha = 0.05



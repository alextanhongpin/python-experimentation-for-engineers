```python
import math

import numpy as np
import scipy.stats as st
import statsmodels.stats.api as sm
```


```python
a = np.zeros(500)
a[:50] = 1
b = np.zeros(600)
b[:90] = 1
```


```python
count1 = a.sum()
nobs1 = a.size
count2 = b.sum()
nobs2 = b.size

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
nobs1 = math.ceil(
    sm.samplesize_proportions_2indep_onetail(
        p2 - p1, p1, 0.8, alternative="two-sided", ratio=1
    )
)
nobs1
```




    686




```python
nobs2 = nobs1
count1 = math.ceil(nobs1 * p1)
count2 = math.ceil(nobs2 * p2)
```


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
    statistic = -2.7654436817399235
    pvalue = 0.005684541990262928
    compare = 'diff'
    method = 'agresti-caffo'
    diff = -0.049562682215743434
    ratio = 0.6699029126213593
    odds_ratio = 0.6329876791867949
    variance = 0.0003193385291546656
    alternative = 'two-sided'
    value = 0
    tuple = (-2.7654436817399235, 0.005684541990262928)




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
    power = 0.8002318546957171
    p_pooled = 0.125
    std_null = 0.46770717334674267
    std_alt = 0.4663689526544408
    nobs1 = 686
    nobs2 = 686.0
    nobs_ratio = 1.0
    alpha = 0.05




```python
sm.zt_ind_solve_power(
    sm.proportion_effectsize(p1, p2), nobs1=None, alpha=0.05, power=0.8
)
```




    680.3526619127882



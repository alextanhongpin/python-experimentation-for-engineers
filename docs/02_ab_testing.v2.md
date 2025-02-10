```python
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import statsmodels.stats.api as sm
import statsmodels.stats.power as sp

plt.rc("figure", figsize=(16, 10))
```


```python
def plot_normal(mu=0, variance=1, label=""):
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, st.norm.pdf(x, mu, sigma), label=label)
```


```python
mu = 0
variance = 1
plot_normal(mu, variance)
```


    
![png](02_ab_testing.v2_files/02_ab_testing.v2_2_0.png)
    



```python
class Money:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "${:.2f}".format(self.value)


class Exchange:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost

    def sample(self) -> int:
        return self.cost + np.random.normal()


class TimeOfDayEffectWrapper:
    def __init__(self, exchange: Exchange):
        self.exchange = exchange

    def sample(self, tod="afternoon"):
        bias = 2.5 if tod == "morning" else 0.0
        return self.exchange.sample()


asdaq = Exchange(name="ASDAQ", cost=12)
print(Money(asdaq.sample()))

asdaq = TimeOfDayEffectWrapper(asdaq)
print(Money(asdaq.sample("morning")))
print(Money(asdaq.sample("afternoon")))

byse = Exchange(name="BYSE", cost=10)
byse = TimeOfDayEffectWrapper(byse)
print(Money(byse.sample()))
```

    $11.29
    $12.37
    $10.59
    $11.06



```python
print(Money(np.mean([byse.sample() for _ in range(100)])))
print(Money(np.mean([asdaq.sample() for _ in range(100)])))
```

    $10.03
    $11.95



```python
def aggregate_measurements(n, exchange: Exchange):
    return np.mean([exchange.sample() for _ in range(n)])


print(np.std([aggregate_measurements(10, asdaq) for _ in range(1000)]))
print(np.std([aggregate_measurements(100, asdaq) for _ in range(1000)]))
print(np.std([aggregate_measurements(1000, asdaq) for _ in range(1000)]))
```

    0.31478235874225285
    0.09838429118781078
    0.03192415473172491



```python
sample_size = 10
sample_count = 1000

sample_population = np.array(
    [aggregate_measurements(sample_size, asdaq) for _ in range(sample_count)]
)
print("Mean: {:.4f}".format(sample_population.mean()))
print("Std Error: {:.4f}".format(st.sem(sample_population)))
print("Std Deviation: {:.4f}".format(sample_population.std()))
```

    Mean: 12.0138
    Std Error: 0.0099
    Std Deviation: 0.3120



```python
plt.hist(sample_population)
plt.title("Normal distribution");
```


    
![png](02_ab_testing.v2_files/02_ab_testing.v2_7_0.png)
    



```python
asdaq_sample = np.array(
    [aggregate_measurements(sample_size, asdaq) for _ in range(sample_count)]
)
byse_sample = np.array(
    [aggregate_measurements(sample_size, byse) for _ in range(sample_count)]
)
plt.hist(asdaq_sample, label="asdaq", alpha=0.9)
plt.hist(byse_sample, label="byse", alpha=0.9)
plt.legend();
```


    
![png](02_ab_testing.v2_files/02_ab_testing.v2_8_0.png)
    


The more average measurements we take, the smaller the standard deviation.


```python
alpha = 0.05
beta = 0.2

diff = 1
power = 1 - beta
n1 = sp.normal_sample_size_one_tail(
    diff, power, alpha, std_null=1, std_alternative=None
)
n1 = int(np.ceil(n1))
print("at least {} individual measurements are needed".format(n1))
```

    at least 7 individual measurements are needed



```python
n = 30
asdaq_samples = []
byse_samples = []

# Randomly collecting samples for both ASDAQ and NYSE.
while len(asdaq_samples) < n or len(byse_samples) < n:
    if np.random.random() < 0.5:
        asdaq_samples.append(asdaq.sample())
    else:
        byse_samples.append(byse.sample())

plt.boxplot(
    [asdaq_samples, byse_samples],
    tick_labels=["ASDAQ", "BYSE"],
    showmeans=True,
    meanline=True,
);
```


    
![png](02_ab_testing.v2_files/02_ab_testing.v2_11_0.png)
    



```python
sem = st.sem(asdaq_samples)
std = np.std(asdaq_samples)
mean = np.mean(asdaq_samples)
print("sem={:.4f} | std={:.4f} | mean={:.4f}".format(sem, std, mean))
print(len(asdaq_samples))
```

    sem=0.1864 | std=1.0040 | mean=11.9427
    30



```python
sem = st.sem(byse_samples)
std = np.std(byse_samples)
mean = np.mean(byse_samples)
print("sem={:.4f} | std={:.4f} | mean={:.4f}".format(sem, std, mean))
print(len(byse_samples))
```

    sem=0.1474 | std=0.9438 | mean=10.1549
    42



```python
plot_normal(np.mean(asdaq_samples), np.var(asdaq_samples), label="ASDAQ")
plot_normal(np.mean(byse_samples), np.var(byse_samples), label="BYSE")
plt.legend();
```


    
![png](02_ab_testing.v2_files/02_ab_testing.v2_14_0.png)
    



```python
# NOTE: The position of the X values are important.
# X1 - X2, for smaller alternative.
tstat, pvalue = sm.ztest(
    asdaq_samples,  # Control, A
    byse_samples,  # Treatment, B
    usevar="unequal",
    alternative="larger",
)
tstat, pvalue, pvalue < 0.05, st.norm.sf(tstat)
```




    (7.522193699003272, 2.693235066418762e-14, True, 2.693235066418762e-14)




```python
# correct if the population S.D. is expected to be equal for the two groups.
def cohens_d(x1, x2):
    """
    Specify the division argument on the variance with ddof=1 into the std function,
    i.e. numpy.std(c0, ddof=1).
    numpy's standard deviation default behaviour is to divide by n,
    whereas with ddof=1 it will divide by n-1.
    """
    n1 = len(x1)
    n2 = len(x2)
    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)
    # We can also operate under the assumption that the standard deviation of s2 = s1

    # Difference in mean
    u = np.mean(x1) - np.mean(x2)
    # Pooled standard deviation.
    s = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return u / s


es = cohens_d(asdaq_samples, byse_samples)
es
```




    1.818525393844541




```python
import statistics

statistics.stdev(asdaq_samples), np.std(asdaq_samples, ddof=1)
```




    (1.0211818137634798, 1.0211818137634798)




```python
es, var_ = sm.effectsize_smd(
    # Treatment.
    np.mean(byse_samples),
    np.std(byse_samples, ddof=1),
    len(byse_samples),
    # Control.
    np.mean(asdaq_samples),
    np.std(asdaq_samples, ddof=1),
    len(asdaq_samples),
)
es = np.abs(es)
es, var_
```




    (1.7989713573515889, 0.08091818732630864)




```python
# es = 1 / 1
alpha = 0.05
power = 0.8
sm.NormalIndPower().solve_power(
    effect_size=es,
    nobs1=None,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative="larger",
)
```




    3.820757397516575




```python
sm.zt_ind_solve_power(
    effect_size=es,
    nobs1=None,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative="larger",
)
```




    3.820757397516575




```python
sm.tt_ind_solve_power(
    effect_size=es,
    nobs1=None,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative="larger",
)
```




    4.677329390106186



If both sample has the same size, the pooled variance is simple the average:


```python
def sample_size(x1, x2, alpha=0.05, beta=0.2, kappa=1.0):
    # Sampling ratio, κ = n_1 / n_2
    # Type I error rate, α
    # Type II error rate, β
    # Mean of group 1, μ_1 and Mean of group 2, μ_2
    u1, u2 = np.mean(x1), np.mean(x2)
    # Sample standard deviation of group 1 and Sample standard deviation of group 2
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    # We can also operate under the assumption that the standard deviation is the same.
    # s2 = s1

    n1 = (s1**2 + s2**2 / kappa) * (
        (st.norm.ppf(1 - alpha) + st.norm.ppf(1 - beta)) / (u1 - u2)
    ) ** 2
    return n1


sample_size(asdaq_samples, byse_samples)
```




    3.7822460278450882




```python
def analyze(x1, x2):
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    u1, u2 = np.mean(x1), np.mean(x2)
    n1, n2 = len(x1), len(x2)

    # If the variance are equal, we can use the pooled variance.
    # pooled_var = ((n_1 - 1) * s_1**2 + (n_2 - 1) * s_2**2) / (n_1 - 1 + n_2 - 1)
    u = u1 - u2
    s = np.sqrt(s1**2 / n1 + s2**2 / n2)
    z = u / s
    p = st.norm.sf(z)
    return z, p


analyze(asdaq_samples, byse_samples)
```




    (7.522193699003273, 2.6932350664187316e-14)




```python
x1 = asdaq_samples
x2 = byse_samples
sm.ttest_ind(
    x1, x2, alternative="larger", usevar="unequal", weights=(None, None), value=0
)
```




    (7.522193699003272, 1.6041495447359274e-10, 60.00114424536958)




```python
## Validating

# Number of individual samples to collect.
n = 3
test_asdaq_samples = np.random.choice(asdaq_samples, n)
test_byse_samples = np.random.choice(byse_samples, n)
```


```python
tstat, pvalue = sm.ztest(
    test_asdaq_samples,  # Control, A
    test_byse_samples,  # Treatment, B
    usevar="unequal",
    alternative="larger",
)
print("z-score:", np.round(tstat, 4))
print("p-value:", np.round(pvalue, 4))
print("reject h0:", pvalue < 0.05)
```

    z-score: 2.5535
    p-value: 0.0053
    reject h0: True



```python
tstat, pvalue, df = sm.ttest_ind(
    asdaq_samples,  # Control, A
    byse_samples,  # Treatment, B
    usevar="unequal",
    alternative="larger",
)
print("z-score:", np.round(tstat, 4))
print("p-value:", np.round(pvalue, 4))
print("df:", df)
print("reject h0:", pvalue < 0.05)
```

    z-score: 7.5222
    p-value: 0.0
    df: 60.00114424536958
    reject h0: True



```python
tstat, pvalue, df = sm.ttest_ind(
    test_asdaq_samples,  # Control, A
    test_byse_samples,  # Treatment, B
    usevar="unequal",
    alternative="larger",
)
print("z-score:", np.round(tstat, 4))
print("p-value:", np.round(pvalue, 4))
print("df:", df)
print("reject h0:", pvalue < 0.05)
```

    z-score: 2.5535
    p-value: 0.0513
    df: 2.44513487789165
    reject h0: False



```python
st.ttest_ind(test_asdaq_samples, test_byse_samples)
```




    TtestResult(statistic=2.55353718605159, pvalue=0.06306844114724655, df=4.0)




```python
n = 3

test_asdaq_samples = np.random.choice(asdaq_samples, n)
# Assuming we use the same samples as ASDAQ, just different measurements.
test_byse_samples = np.random.choice(asdaq_samples, n)
tstat, pvalue = sm.ztest(
    test_asdaq_samples,  # Control, A
    test_byse_samples,  # Treatment, B
    usevar="unequal",
    alternative="larger",
)
print("z-score:", np.round(tstat, 4))
print("p-value:", np.round(pvalue, 4))
print("reject h0:", pvalue < 0.05)
```

    z-score: -0.9057
    p-value: 0.8175
    reject h0: False



```python
st.ttest_ind(test_asdaq_samples, test_byse_samples)
```




    TtestResult(statistic=-0.9057427717065338, pvalue=0.416296560860438, df=4.0)




```python
st.ttest_ind(np.random.choice(asdaq_samples, 10), np.random.choice(asdaq_samples, 10))
```




    TtestResult(statistic=-1.0493027767393903, pvalue=0.307923939649284, df=18.0)



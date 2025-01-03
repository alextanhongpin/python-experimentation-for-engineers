# A/B testing: Evaluating a modification to your system

Three steps in A/B testing:
- design: how many measurements to take
- measure: use randomization to measure
- analyze: compare the busines metric estimates of A and B

## Simulate the trading system


```python
import numpy as np


def trading_system(exchange):
    if exchange == "ASDAQ":
        execution_cost = 12
    elif exchange == "BYSE":
        execution_cost = 10
    else:
        raise ValueError("unknown exchange: {}".format(exchange))
    execution_cost += np.random.normal()
    return execution_cost
```


```python
np.random.seed(17)

trading_system("ASDAQ")
```




    12.27626589002132



## Compare execution cost


```python
np.random.seed(17)

print(trading_system("ASDAQ"))
print(trading_system("BYSE"))
```

    12.27626589002132
    8.145371921193496


Observation: it is cheaper to trade on BYSE


```python
np.random.seed(18)

print(trading_system("ASDAQ"))
print(trading_system("BYSE"))
```

    12.079428443806204
    12.190202357414222


Observation: it is now cheaper to trade on ASDAQ. The measure value varies from measurement to measurement.

### Variation


```python
np.random.seed(17)

print(np.array([trading_system("ASDAQ") for _ in range(100)]).mean())
print(np.array([trading_system("BYSE") for _ in range(100)]).mean())
```

    12.111509794247766
    10.008382946497413



```python
print(np.array([trading_system("ASDAQ") for _ in range(100)]).mean())
print(np.array([trading_system("BYSE") for _ in range(100)]).mean())
```

    11.880880186907996
    9.99591773728191


### Bias

Below is a simulator that accounts for time of day (tod).


```python
def trading_system_tod(exchange, time_of_day):
    if time_of_day == "morning":
        bias = 2.5
    elif time_of_day == "afternoon":
        bias = 0
    else:
        raise ValueError("unknown time of day: {}".format(time_of_day))
    return bias + trading_system(exchange)
```


```python
np.random.seed(17)

print(np.array([trading_system_tod("ASDAQ", "morning") for _ in range(100)]).mean())
print(np.array([trading_system_tod("ASDAQ", "afternoon") for _ in range(100)]).mean())
```

    14.611509794247766
    12.008382946497411


Observation: it is cheaper to trade in the afternoon.


```python
np.random.seed(17)

print(np.array([trading_system_tod("BYSE", "morning") for _ in range(100)]).mean())
print(np.array([trading_system_tod("ASDAQ", "afternoon") for _ in range(100)]).mean())
```

    12.611509794247766
    12.008382946497411


Observation: ASDAQ is more expensive, but it would appear as if it is cheaper than BYSE if traded during the afternoon.

### Randomization

To remove __confounder bias__ (when a bias is applied differently and consistently), we apply __randomization__ when taking measurements.


```python
def randomized_measurement():
    asdaq_measurement = []
    byse_measurement = []
    for tod in ["morning", "afternoon"]:
        for _ in range(100):
            if np.random.randint(2) == 0:
                asdaq_measurement.append(trading_system_tod("ASDAQ", tod))
            else:
                byse_measurement.append(trading_system_tod("BYSE", tod))
    return np.array(asdaq_measurement).mean(), np.array(byse_measurement).mean()
```


```python
np.random.seed(17)

randomized_measurement()
```




    (13.39588870623852, 11.259639285763223)



## Take a precise measurement

### Mitigate measurement variation with replication


```python
np.random.seed(17)

measurements = np.array([trading_system("ASDAQ") for _ in range(3)])
measurements
```




    array([12.27626589, 10.14537192, 12.62390111])




```python
measurements.mean()
```




    11.681846307513723




```python
measurements - 12
```




    array([ 0.27626589, -1.85462808,  0.62390111])




```python
measurements.mean() - 12
```




    -0.3181536924862769




```python
def aggregate_measurement(exchange, num_individual_measurements):
    individual_measurements = np.array(
        [trading_system(exchange) for _ in range(num_individual_measurements)]
    )
    return individual_measurements.mean()
```


```python
np.random.seed(17)

print(aggregate_measurement("ASDAQ", 300))
print(aggregate_measurement("BYSE", 300))
```

    12.000257642551059
    10.051095649188758



```python
print(aggregate_measurement("ASDAQ", 300))
print(aggregate_measurement("BYSE", 300))
```

    11.987318214094266
    10.021053044438455


### Standard Error


```python
agg_3 = np.array([aggregate_measurement("ASDAQ", 3) for _ in range(1000)])
agg_30 = np.array([aggregate_measurement("ASDAQ", 30) for _ in range(1000)])
agg_300 = np.array([aggregate_measurement("ASDAQ", 300) for _ in range(1000)])

agg_3.std(), agg_30.std(), agg_300.std()
```




    (0.5778543829446465, 0.1794924850151226, 0.058012150188856464)



Observation: the standard deviation decreases as the number of individual measurements in each aggregate measurement increases.


```python
def standard_error(measurements):
    return measurements.std() / np.sqrt(len(measurements))
```


```python
def aggregate_measurement_with_se(exchange, num_individual_measurements):
    individual_measurements = np.array(
        [trading_system(exchange) for _ in range(num_individual_measurements)]
    )
    aggregate_measurement = individual_measurements.mean()
    return aggregate_measurement, standard_error(individual_measurements)
```


```python
np.random.seed(17)

print(aggregate_measurement_with_se("ASDAQ", 300))
print(aggregate_measurement_with_se("BYSE", 300))
```

    (12.000257642551059, 0.060254756364981225)
    (10.051095649188758, 0.05714189794415452)


## Run an A/B test

### Analyze your measurements


```python
np.random.seed(17)

num_individual_measurements = 10
agg_asdaq, se_asdaq = aggregate_measurement_with_se(
    "ASDAQ", num_individual_measurements
)
agg_byse, se_byse = aggregate_measurement_with_se("BYSE", num_individual_measurements)
delta = agg_byse - agg_asdaq
se_delta = np.sqrt(se_byse**2 + se_asdaq**2)
z_score = delta / se_delta
z_score
```




    -4.4851273191475025




```python
import scipy.stats as st

# 90% confidence interval.
print(st.norm.ppf(1 - 0.1))  # z-score from p-value
print(st.norm.cdf(1.64))  # z-score to p-value
print(
    st.norm.sf(abs(1.64)), st.norm.sf(abs(1.64)) * 2
)  # z-score to p-value, sf: survival function is 1-cdf
print()

# 95% confidence interval.
print(st.norm.ppf(1 - 0.05))  # z-score from p-value
print(st.norm.cdf(1.96))  # z-score to p-value
print(st.norm.sf(abs(1.96)) * 2)  # z-score to p-value
print()

# 99% confidence interval.
print(st.norm.ppf(1 - 0.01))
print(st.norm.cdf(2.576))
print(st.norm.sf(abs(2.576)) * 2)  # z-score to p-value
```

    1.2815515655446004
    0.9494974165258963
    0.050502583474103704 0.10100516694820741
    
    1.644853626951472
    0.9750021048517795
    0.04999579029644087
    
    2.3263478740408408
    0.995002467684265
    0.009995064631470029



```python
for z_score in [1, 1.96, 2.48, 5.0]:
    p_value = st.norm.sf(abs(z_score)) * 2  # two-tailed test
    print(p_value)
```

    0.31731050786291415
    0.04999579029644087
    0.013138238271093524
    5.733031437583869e-07


We know we can reject the null hypothesis is the value is below alpha of 0.05%:

```python
1 - scipy.stats.norm.cdf((x-mu)/(std/np.sqrt(n)) < alpha
```
Where
- x = sample mean
- mu = population mean
- std = population standard deviation

We can use it to solve `n`, which is the population mean:

```python
np.sqrt(n) > std * scipy.stats.norm.ppf(1 - alpha) / (x - mu)
```

### Design the A/B test


```python
def ab_test_design(sd_1_delta, practical_significance):
    num_individual_measurements = (1.64 * sd_1_delta / practical_significance) ** 2
    return np.ceil(num_individual_measurements)
```


```python
np.random.seed(17)

sd_1_asdaq = np.array([trading_system("ASDAQ") for _ in range(100)]).std()
sd_1_byse = sd_1_asdaq
sd_1_delta = np.sqrt(sd_1_asdaq**2 + sd_1_byse**2)
practical_significance = 1
ab_test_design(sd_1_delta, practical_significance)
```




    7.0



Observation: If you take seven individual measurements, you'll have a 5% chance of a false positive - of incorrectly acting as if BYSE is better than ASDAQ.

### False Negatives


```python
def ab_test_design_2(sd_1_delta, practical_significance):
    """A/B test design with power analysis"""
    num_individual_measurements = (2.48 * sd_1_delta / practical_significance) ** 2
    return np.ceil(num_individual_measurements)
```


```python
np.random.seed(17)

sd_1_asdaq = np.array([trading_system("ASDAQ") for _ in range(100)]).std()
sd_1_byse = sd_1_asdaq
sd_1_delta = np.sqrt(sd_1_asdaq**2 + sd_1_byse**2)
prac_sig = 1.0
ab_test_design_2(sd_1_delta, prac_sig)
```




    16.0



### Measure and analyze


```python
def measure(min_individual_measurements):
    ind_asdaq = []
    ind_byse = []
    while (
        len(ind_asdaq) < min_individual_measurements
        and len(ind_byse) < min_individual_measurements
    ):
        if np.random.randint(2) == 0:
            ind_asdaq.append(trading_system("ASDAQ"))
        else:
            ind_byse.append(trading_system("BYSE"))
    return np.array(ind_asdaq), np.array(ind_byse)
```


```python
np.random.seed(17)

ind_asdaq, ind_byse = measure(16)
```


```python
ind_byse.mean() - ind_asdaq.mean()
```




    -2.7483767796620846




```python
def analyze(ind_asdaq, ind_byse):
    agg_asdaq = ind_asdaq.mean()
    se_asdaq = ind_asdaq.std() / np.sqrt(len(ind_asdaq))

    agg_byse = ind_byse.mean()
    se_byse = ind_byse.std() / np.sqrt(len(ind_byse))

    delta = agg_byse - agg_asdaq
    se_delta = np.sqrt(se_asdaq**2 + se_byse**2)

    z = delta / se_delta
    return z
```


```python
analyze(ind_asdaq, ind_byse)
```




    -6.353995237966593




```python
def z_score(dist1, dist2):
    assert isinstance(dist1, np.ndarray), "dist1 is not np.ndarray"
    assert isinstance(dist2, np.ndarray), "dist2 is not np.ndarray"

    mean1 = dist1.mean()
    std_err1 = dist1.std() / np.sqrt(len(dist1))

    mean2 = dist2.mean()
    std_err2 = dist2.std() / np.sqrt(len(dist2))

    delta = mean2 - mean1
    std_err_delta = np.sqrt(std_err2**2 + std_err1**2)

    z = delta / std_err_delta
    return z
```


```python
z_score(ind_asdaq, ind_byse)
```




    -6.353995237966593



Observation: because z is well below the threshold of -1.64, this result is statistically significant. BYSE has passed the second test.

## Using scipy


```python
import scipy.stats as st
from statsmodels.stats.weightstats import ztest

tstat, pvalue = ztest(ind_asdaq, ind_byse)
zscore = st.norm.ppf(pvalue)
tstat, pvalue, zscore
```




    (6.202020909921336, 5.574266926940068e-10, -6.0920335108226755)




```python
from statsmodels.stats.power import TTestIndPower, TTestPower

obj = TTestIndPower()
n = obj.solve_power(effect_size=1, alpha=0.05, power=0.8)
n
```




    16.714722572276173



### Recap of A/B test stages

- design: determined the minimum number of individual measurements needed to be able to detect statistical significance. That number was given by $(2.48 * st_1_delta / prac_sig)**2$
- measure: collect the prescribed number of individual measurements, and randomize between variants to remove confounder bias
- analyze: ensure the difference in cost between BYSE and ASDAQ was **practically significant** (`delta <- prac_sig`) and **statistically significant** (`delta/se_delta <- 1.64`)

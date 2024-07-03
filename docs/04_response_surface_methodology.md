# Response Surface Methodology (RSM)

- designed to optimize continuous parameters


## Optimize a single continuous parameter


```python
import numpy as np
```


```python
def markout_profit(threshold):
    cost = 1
    pps = 1
    signal = np.random.normal()
    eps = 2 * np.random.normal()
    if signal > threshold or signal < -threshold:
        profit = pps * np.abs(signal) - cost + eps
    else:
        profit = 0
    return profit
```


```python
np.random.seed(17)
([markout_profit(0.15) for _ in range(10)], [markout_profit(0.5) for _ in range(10)])
```




    ([-4.432990267591691,
      1.9145236904705367,
      3.8104683277910985,
      0,
      0,
      3.475877498753951,
      0.5427136576035457,
      -4.010724960358923,
      -0.6320195582677376,
      -0.4078249827483169],
     [2.0448324885405436,
      0,
      0.06889559859304795,
      0.5325419453659244,
      0.455015020479104,
      -0.29962873534988227,
      0,
      1.7074529797072233,
      0.4983477757546042,
      -3.90661649896564])




```python
def run_experiment(num_ind, thresholds):
    individual_measurements = {threshold: [] for threshold in thresholds}
    done = set()
    while True:
        threshold = np.random.choice(thresholds)
        profit = markout_profit(threshold)
        individual_measurements[threshold].append(profit)
        if len(individual_measurements[threshold]) >= num_ind:
            done.add(threshold)
        if len(done) == len(thresholds):
            break

    aggregate_measurements = []
    standard_errors = []
    for threshold in thresholds:
        ims = np.array(individual_measurements[threshold])
        aggregate_measurements.append(ims.mean())
        standard_errors.append(ims.std() / np.sqrt(len(ims)))

    return (
        aggregate_measurements,
        standard_errors,
    )
```


```python
np.random.seed(17)
thresholds = np.array([0.5, 1.0, 1.5])
aggregate_measurements, standard_errors = run_experiment(15_000, thresholds)
```


```python
aggregate_measurements, standard_errors
```




    ([0.09848496576216006, 0.1711214205711533, 0.13699263220703944],
     [0.013267765976681866, 0.00957041724519083, 0.006796544642991194])




```python
import matplotlib.pyplot as plt
import numpy as np


def boxplot(data, std_err, positions):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot violin plot
    ax.boxplot(data, positions=positions)

    # Add standard error bars
    for i, (x, y, serr) in enumerate(zip(positions, data.flatten(), std_err)):
        ax.errorbar(x, y, yerr=serr, fmt="o", capsize=5, ecolor="black")

    # Set title and labels
    ax.set_title("Box Plot with Standard Errors")
    ax.set_xlabel("threshold")
    ax.set_ylabel("markout_profit")
    return ax


# Sample data
# For boxplot, every group should be an array of measurements,
# that is why we reshape the 1d array of aggregate measurements.
data = np.array(aggregate_measurements).reshape(1, -1)
std_err = standard_errors
positions = thresholds
boxplot(data, std_err, positions);
```


    
![png](04_response_surface_methodology_files/04_response_surface_methodology_8_0.png)
    



```python
# The function performs a polynomial regression,
# specifically fitting a quadratic model to the given data.
# The returned beta array contains the coefficients of the fitted polynomial,
# allowing you to predict y values given new x values using the model:
def linear_regression(thresholds, aggregate_measurements):
    x = thresholds
    y = aggregate_measurements
    X = np.array([np.ones(len(y)), x, x**2]).T
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta
```


```python
beta = linear_regression(thresholds, aggregate_measurements)
beta
```




    array([-0.08091673,  0.46556864, -0.21353049])




```python
# Calculate the predicted y values using the beta coefficients
def interpolate(thresholds, beta):
    xhat = np.arange(thresholds.min(), thresholds.max() + 1e-6, 0.01)
    XHat = np.array([np.ones(len(xhat)), xhat, xhat**2]).T
    yhat = XHat @ beta
    return xhat, yhat
```


```python
def optimize(thresholds, beta):
    xhat, yhat = interpolate(thresholds, beta)
    i = np.where(yhat == yhat.max())[0][0]
    return xhat[i], yhat[i]
```


```python
beta = linear_regression(thresholds, aggregate_measurements)
threshold_opt, estimated_max_profit = optimize(thresholds, beta)
threshold_opt, estimated_max_profit
```




    (1.0900000000000005, 0.17285751361179258)




```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.array(aggregate_measurements).reshape(1, -1)
std_err = standard_errors

ax = boxplot(data, std_err, thresholds)

xhat = np.arange(thresholds.min(), thresholds.max() + 1e-6, 0.01)
XHat = np.array([np.ones(len(xhat)), xhat, xhat**2]).T
yhat = XHat @ beta

ax.plot(xhat, yhat)

max_x = threshold_opt
max_y = estimated_max_profit
# Annotate the maximum point
plt.annotate(
    f"Maximum value: {max_y:.2f}",
    xy=(max_x, max_y),
    xytext=(max_x - 0.01, max_y - 0.01),
    arrowprops=dict(facecolor="red", shrink=0.05),
)

# Alternatively, we can just use scatter to indicate the maximum point.
plt.scatter(max_x, max_y, color="red", marker="o")


# Show plot
plt.show()
```


    
![png](04_response_surface_methodology_files/04_response_surface_methodology_14_0.png)
    



```python
np.random.seed(17)
aggregate_measurement, standard_error = run_experiment(15_000, [threshold_opt])
(
    aggregate_measurement[0] - 2 * standard_error[0],
    aggregate_measurement[0] + 2 * standard_error[0],
)
```




    (0.14048962175141153, 0.17627270610659548)



## Optimizing two or more continuous parameters


```python
def markout_profit_2D(threshold, order_size):
    cost = 1
    pps = 1  # Profit per signal

    # Adverse selection cost.
    asc = 0.001 * np.exp(2 * order_size)
    signal = np.random.normal()
    eps = 2 * np.random.normal()

    # Buy when signal is strong positive,
    # sell when the signal is strong negative.
    if signal > threshold or signal < -threshold:
        # Profit is offset by adverse selection.
        profit = order_size * (pps * np.abs(signal) - cost + eps) - asc
    else:
        # There is no profit if we don't trade.
        profit = 0
    return profit
```


```python
# Face-centered central composite design
def design_ccd(thresholds, order_sizes):
    parameters = [
        (threshold, order_size)
        for threshold in thresholds
        for order_size in order_sizes
    ]
    return parameters
```


```python
parameters = design_ccd(thresholds=[0.5, 1.0, 1.5], order_sizes=[1, 1.5, 2])
parameters
```




    [(0.5, 1),
     (0.5, 1.5),
     (0.5, 2),
     (1.0, 1),
     (1.0, 1.5),
     (1.0, 2),
     (1.5, 1),
     (1.5, 1.5),
     (1.5, 2)]




```python
import random


def run_experiment_2D(num_ind, parameters):
    individual_measurements = {parameter: [] for parameter in parameters}
    done = set()
    while True:
        parameter = random.choice(parameters)
        threshold, order_size = parameter
        profit = markout_profit_2D(threshold, order_size)
        individual_measurements[parameter].append(profit)
        if len(individual_measurements[parameter]) >= num_ind:
            done.add(parameter)
        if len(done) == len(individual_measurements):
            break

    aggregate_measurements = []
    standard_errors = []
    for parameter in parameters:
        ims = np.array(individual_measurements[parameter])
        aggregate_measurements.append(ims.mean())
        standard_errors.append(ims.std() / np.sqrt(len(ims)))

    return aggregate_measurements, standard_errors
```


```python
np.random.seed(17)
# parameters = design_ccd(thresholds=[1, 1.5, 2], order_sizes=[1, 1.5, 2])
# parameters = design_ccd(thresholds=[0.5, 1, 1.5], order_sizes=[2.5, 3.0, 3.5])
parameters = design_ccd(thresholds=[0.75, 1, 1.25], order_sizes=[2.75, 3.0, 3.25])

aggregate_measurements, standard_errors = run_experiment_2D(15_000, parameters)
aggregate_measurements, standard_errors
```




    ([0.30501820887686837,
      0.28691078325014896,
      0.19626465227181056,
      0.372181067940838,
      0.346952909305505,
      0.30607233748792045,
      0.4030158914268578,
      0.36738006900560755,
      0.3452359914218985],
     [0.03077800980184445,
      0.034334278171763155,
      0.03638664994304799,
      0.02589157754775616,
      0.02833935463895004,
      0.029985599966878166,
      0.021855672345450234,
      0.023858778916175757,
      0.02526222515675146])




```python
data = np.array(aggregate_measurements).reshape(1, -1)
std_err = standard_errors
positions = [i[0] * 10 + i[1] for i in parameters]
boxplot(data, std_err, positions)
```




    <Axes: title={'center': 'Box Plot with Standard Errors'}, xlabel='threshold', ylabel='markout_profit'>




    
![png](04_response_surface_methodology_files/04_response_surface_methodology_22_1.png)
    


### Linear regression for two parameters


```python
def linear_regression_2D(parameters, aggregate_measurements):
    parameters = np.array(parameters)
    x0 = parameters[:, 0]
    x1 = parameters[:, 1]
    y = aggregate_measurements
    X = np.array([np.ones(len(y)), x0, x1, x0**2, x1**2, x0 * x1]).T
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta
```


```python
beta = linear_regression_2D(parameters, aggregate_measurements)
beta
```




    array([-0.97497231,  0.38840584,  0.83620056, -0.39089876, -0.19919833,
            0.20389463])




```python
def interpolate_2D(parameters, beta):
    parameters = np.array(parameters)
    x0_values = np.arange(parameters[:, 0].min(), parameters[:, 0].max() + 1e-6, 0.01)
    x1_values = np.arange(parameters[:, 1].min(), parameters[:, 1].max() + 1e-6, 0.01)
    x0hat_2d, x1hat_2d = np.meshgrid(x0_values, x1_values)
    x0hat = x0hat_2d.flatten()
    x1hat = x1hat_2d.flatten()
    XHat = np.array(
        [np.ones(len(x0hat)), x0hat, x1hat, x0hat**2, x1hat**2, x0hat * x1hat]
    ).T
    yhat = XHat @ beta
    yhat_2d = np.reshape(yhat, (len(x1_values), len(x0_values)))
    return x0hat_2d, x1hat_2d, yhat_2d
```


```python
def optimize_2D(parameters, beta):
    x0hat, x1hat, yhat = interpolate_2D(parameters, beta)
    i = np.where(yhat == yhat.max())
    return x0hat[i][0], x1hat[i][0], yhat[i][0]
```


```python
beta = linear_regression_2D(parameters, aggregate_measurements)
threshold_opt, order_size_opt, estimated_max_profit = optimize_2D(parameters, beta)
threshold_opt, order_size_opt, estimated_max_profit
```




    (1.2100000000000004, 2.75, 0.39425743094342747)




```python
x0hat, x1hat, yhat = interpolate_2D(parameters, beta)
```


```python
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# http://en.wikipedia.org/wiki/File:Bilininterp.png
xi = x0hat
yi = x1hat
zi = yhat


# I want 20 "levels" to be shown
contour_breaks = 20
ticks = np.linspace(zi.min(), zi.max(), contour_breaks, endpoint=True)

fig = plt.figure()
axes = fig.add_subplot(111, aspect="equal")
axes.contour(xi, yi, zi, ticks[1:-1], colors="k")
fill = axes.contourf(xi, yi, zi, ticks, cmap=cm.jet)
fig.colorbar(fill, ticks=ticks)

# Show the plots
plt.show()
```


    
![png](04_response_surface_methodology_files/04_response_surface_methodology_30_0.png)
    



```python
data = np.array(aggregate_measurements).reshape(1, -1)
std_err = standard_errors
positions = [i[0] * 10 + i[1] for i in parameters]
boxplot(data, std_err, positions)
```




    <Axes: title={'center': 'Box Plot with Standard Errors'}, xlabel='threshold', ylabel='markout_profit'>




    
![png](04_response_surface_methodology_files/04_response_surface_methodology_31_1.png)
    



```python
aggregate_measurements, standard_error = run_experiment_2D(
    15_000, parameters=[(threshold_opt, order_size_opt)]
)
aggregate_measurements, standard_error
```




    ([0.36638808798987776], [0.02212615846153084])




```python
(
    aggregate_measurement[0] - 2 * standard_error[0],
    aggregate_measurement[0] + 2 * standard_error[0],
)
```




    (0.11412884700594182, 0.20263348085206517)



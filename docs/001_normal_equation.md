# Solving Linear Equation

https://www.eecis.udel.edu/~boncelet/ipynb/LR_NYC_Example.html


```python
import numpy as np
```


```python
np.c_[[1, 2], [3, 4]]
```




    array([[1, 3],
           [2, 4]])




```python
np.stack([[1, 2], [3, 4]], axis=1)
```




    array([[1, 3],
           [2, 4]])




```python
X = 2 * np.random.rand(100)
y = 4 + 3 * X + np.random.randn(100)
```

## Solving the normal equation manually


```python
# Add 1 as bias term for x0
X_b = np.stack([np.ones(100), X], axis=1)
theta = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)
theta
```




    array([4.12802768, 2.72259853])




```python
X_test = [1, 5]
X_test @ theta
```




    17.741020326494528




```python
X_b = np.stack([np.ones(X.shape[0]), X], axis=1)
theta, residuals, rank, s = np.linalg.lstsq(X_b, y)
theta
```

    /var/folders/v5/8v9k6wcn65jbbct8spl3wwsh0000gn/T/ipykernel_65925/1142066073.py:2: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      theta, residuals, rank, s = np.linalg.lstsq(X_b, y)





    array([4.12802768, 2.72259853])




```python
X_test @ theta
```




    17.74102032649454




```python
theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
theta
```




    array([4.12802768, 2.72259853])




```python
X_test @ theta
```




    17.741020326494507



## Using statsmodels


```python
import statsmodels.api as sm

model = sm.OLS(y, sm.add_constant(X))
results = model.fit()
results.params
```




    array([4.12802768, 2.72259853])




```python
model.predict(results.params, exog=X_test)
```




    17.741020326494535



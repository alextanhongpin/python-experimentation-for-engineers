# Sample Size for experiment


Hypothesis testing is a formal way of checking if a hypothesis about a population is true or not.


A population proportion is the share of a population that belongs to a particular category. Hypothesis tests are used to check a claim about the size of that population proportion. Normally used for A/B testing.

- https://www.kaggle.com/code/kuixizhu/starbucks-promotion-strategy-analysis
- https://www.w3schools.com/statistics/statistics_hypothesis_testing_mean.php

The formula for variance is:


$\sigma^2 = \frac{\sum_{i=1}^{N}\cdot(x_i - \mu)^2}{n}$

Standard deviation is just the square root of variance:

$\sigma = \sqrt{\frac{\sum_{i=1}^{N}\cdot(x_i - \mu)^2}{n}}$


We can find the z-score by taking the difference in mean divided by delta standard error:

$z = \frac{\mu_1-\mu_2}{se_d}$

Where standard error delta is calculated by:


$se_d = \sqrt{se_1^2+se_2^2} = \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}$

$se_n = \frac{\sigma_n}{\sqrt{n_n}}$

If both n is the same, then we can simplify standard error delta to:

$se_d = \sqrt{\frac{\sigma_1^2+\sigma_2^2}{n}}$

If we want to calculate sample size for one observation, given that both is the same size, solving the equation above gives us:

$n_1 = (\sigma_1^2+\frac{\sigma_2^2}{k})\cdot(\frac{z_{1-\alpha} + z_{1-\beta}}{\mu_1-\mu_2})^2$

K is the ratio:

$n_2 = k\cdot{n_1}$
When calculating sample size, the approach differs depending on whether you are calculating the sample size for a mean difference or for proportions. Below are the methods for both:

### Sample Size Calculation for Mean Difference
To calculate the sample size for comparing the means of two groups, you can use the following formula:

\[ n = \left( \frac{{Z_{\alpha/2} + Z_{\beta}}{ \sigma_d }}{ \Delta } \right)^2 \]

Where:
- \( n \) = required sample size per group
- \( Z_{\alpha/2} \) = z-value for the desired level of confidence (e.g., 1.96 for 95% confidence)
- \( Z_{\beta} \) = z-value for the desired power (e.g., 0.84 for 80% power)
- \( \sigma_d \) = standard deviation of the difference between the two means
- \( \Delta \) = the difference between the two means you want to detect

### Sample Size Calculation for Proportions
To calculate the sample size for comparing the proportions of two groups, you can use the following formula:

\[ n = \frac{{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1 - p_1) + p_2(1 - p_2))}}{(p_1 - p_2)^2} \]

Where:
- \( n \) = required sample size per group
- \( Z_{\alpha/2} \) = z-value for the desired level of confidence (e.g., 1.96 for 95% confidence)
- \( Z_{\beta} \) = z-value for the desired power (e.g., 0.84 for 80% power)
- \( p_1 \) = proportion in group 1
- \( p_2 \) = proportion in group 2
- \( p_1 - p_2 \) = the difference between the two proportions you want to detect

### Example Calculations

#### Mean Difference Example
Let's say you want to detect a mean difference of 5 units, with a standard deviation of 10 units, 95% confidence level (Z = 1.96), and 80% power (Z = 0.84).

\[ n = \left( \frac{(1.96 + 0.84) \cdot 10}{5} \right)^2 \]
\[ n = \left( \frac{2.8 \cdot 10}{5} \right)^2 \]
\[ n = (5.6)^2 \]
\[ n = 31.36 \]

So, you would need approximately 32 participants per group.

#### Proportion Example
Let's say you want to detect a difference between proportions of 0.1 (10%), with proportions in each group being 0.5 (50%), 95% confidence level (Z = 1.96), and 80% power (Z = 0.84).

\[ n = \frac{(1.96 + 0.84)^2 \cdot (0.5 \cdot (1 - 0.5) + 0.5 \cdot (1 - 0.5))}{(0.1)^2} \]
\[ n = \frac{(2.8)^2 \cdot (0.25 + 0.25)}{0.01} \]
\[ n = \frac{7.84 \cdot 0.5}{0.01} \]
\[ n = \frac{3.92}{0.01} \]
\[ n = 392 \]

So, you would need approximately 392 participants per group.

These formulas and examples should help you calculate the sample size for your study based on the type of data you are working with.

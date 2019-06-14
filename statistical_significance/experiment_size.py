import numpy as np
import scipy.stats as stats

"""
Compute the minimum number of samples needed to achieve a desired power
level for a given effect size.

Input parameters:
    p_null: base success rate under null hypothesis
    p_alt : desired success rate to be detected
    alpha : Type-I error rate
    beta  : Type-II error rate

Output value:
    n : Number of samples required for each group to obtain desired power
"""
p_null = 0.10
p_alt = 0.12

alpha = .05
beta = .20

# Get necessary z-scores and standard deviations (@ 1 obs per group)
z_null = stats.norm.ppf(1 - alpha)
z_alt = stats.norm.ppf(beta)
sd_null = np.sqrt(p_null * (1 - p_null) + p_null * (1 - p_null))
sd_alt = np.sqrt(p_null * (1 - p_null) + p_alt * (1 - p_alt))

# Compute and return minimum sample size
p_diff = p_alt - p_null
n = ((z_null * sd_null - z_alt * sd_alt) / p_diff) ** 2

print(n)
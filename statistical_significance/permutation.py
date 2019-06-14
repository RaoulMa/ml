import numpy as np
import pandas as pd

"""
Compute a confidence interval for a quantile of a dataset using a bootstrap
method.

Input parameters:
    x: 1-D array-like of data for independent / grouping feature as 0s and 1s
    y: 1-D array-like of data for dependent / output feature
    q: quantile to be estimated, must be between 0 and 1
    alternative: type of test to perform, {'less', 'greater'}
    n_trials: number of permutation trials to perform

Output value:
    p: estimated p-value of test
"""

data = pd.read_csv('permutation_data.csv')

x = data['time']
y = data['condition']
q = 0.9
alternative = 'less'

n_trials = 1000

# initialize storage of bootstrapped sample quantiles
sample_diffs = []

# For each trial...
for _ in range(n_trials):
    # randomly permute the grouping labels
    labels = np.random.permutation(y)

    # compute the difference in quantiles
    cond_q = np.percentile(x[labels == 0], 100 * q)
    exp_q = np.percentile(x[labels == 1], 100 * q)

    # and add the value to the list of sampled differences
    sample_diffs.append(exp_q - cond_q)

# compute observed statistic
cond_q = np.percentile(x[y == 0], 100 * q)
exp_q = np.percentile(x[y == 1], 100 * q)
obs_diff = exp_q - cond_q

# compute a p-value
if alternative == 'less':
    hits = (sample_diffs <= obs_diff).sum()
elif alternative == 'greater':
    hits = (sample_diffs >= obs_diff).sum()

p_value = hits / n_trials

print(p_value)
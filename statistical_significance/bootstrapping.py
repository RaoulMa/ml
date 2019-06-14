import numpy as np
import pandas as pd

"""
 Compute a confidence interval for a quantile of a dataset using a bootstrap
 method.

 Input parameters:
     data: data in form of 1-D array-like (e.g. numpy array or Pandas series)
     q: quantile to be estimated, must be between 0 and 1
     c: confidence interval width
     n_trials: number of bootstrap samples to perform

 Output value:
     ci: Tuple indicating lower and upper bounds of bootstrapped
         confidence interval
 """

data = pd.read_csv('bootstrapping_data.csv')

print(data.head())

data = data['time']

c = .95
n_trials = 1000
q = 0.9

# initialize storage of bootstrapped sample quantiles
n_points = data.shape[0]
sample_qs = []

# For each trial...
for _ in range(n_trials):
    # draw a random sample from the data with replacement...
    sample = np.random.choice(data, n_points, replace=True)

    # compute the desired quantile...
    sample_q = np.percentile(sample, 100 * q)

    # and add the value to the list of sampled quantiles
    sample_qs.append(sample_q)

# Compute the confidence interval bounds
lower_limit = np.percentile(sample_qs, (1 - c) / 2 * 100)
upper_limit = np.percentile(sample_qs, (1 + c) / 2 * 100)

print(lower_limit, upper_limit)
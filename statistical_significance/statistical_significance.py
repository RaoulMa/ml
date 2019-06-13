import numpy as np
import pandas as pd
import scipy.stats as stats

# The data set contains users split into a control and experimental group.
# Each user visits a website and maybe clicks on the download button (1)
# or not (0)

data = pd.read_csv('statistical_significance_data.csv')
print(data.head())

# number of visitors
n_visits = data.shape[0]

# number of visitors in control group
n_control = data.groupby('condition').size()[0]

### check the invariant metric first, i.e. that the control group size is compatible wit the null hypothesis
print('Check the invariant metric')

# null hypothesis
p_null = 0.5

# 1. Solution via Simulation
n_trials = 200000
samples = np.random.binomial(n_visits, p_null, n_trials)
p_value = np.logical_or(samples <= n_control, samples >= (n_visits - n_control)).mean()
print('p-value {:.3f}'.format(p_value))

# 2. Analytic solution via Standardisation
exp = p_null * n_visits
sd = np.sqrt(p_null * (1-p_null) * n_visits)
z = (n_control - exp) / sd
p_value = 2 * stats.norm.cdf(z)
print('p-value {:.3f}'.format(p_value))

### Check the evaluation metric, i.e. that the click rate of the experimental group is significantly higher than
# the click rate of the control group
print('Check the evaluation metric')

# null hypothesis
p_null = data['click'].mean()

# size of control and experimental group
n_control = data.groupby('condition').size()[0]
n_exp = data.groupby('condition').size()[1]

# click probabilit of control and exp groups
p_click = data.groupby('condition').mean()['click']
p_control_click = p_click[0]
p_exp_click = p_click[1]

# ratio R as random variable
R = p_exp_click - p_control_click

# 1. Simulation Solution
n_trials = 20000
control_clicks = np.random.binomial(n_control, p_null, n_trials)
exp_clicks = np.random.binomial(n_exp, p_null, n_trials)
samples = exp_clicks/n_exp - control_clicks/n_control
p_value = (samples >= R).mean()
print('p-value {:.3f}'.format(p_value))

# 2. Analytic Solution
sd = np.sqrt(p_null * (1-p_null) * (1/n_control + 1/n_exp))
z = R / sd
p_value = 1 - stats.norm.cdf(z)
print('p-value {:.3f}'.format(p_value))








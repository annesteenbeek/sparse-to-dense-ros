
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

from scipy import stats


sparse = np.load('dev/sparse_samples.npy')
target = np.load('dev/target_samples.npy')

max_depth = 5
 
valid_mask = target < max_depth
 
sparse = sparse[valid_mask]
target = target[valid_mask]

error = target-sparse

error_norm = error / np.abs(np.sum(error))

rel_error = error/target

ae, loce, scalee = stats.skewnorm.fit(error)
skew_sample = stats.skewnorm(ae, loce, scalee).rvs(len(error))

mean, std = stats.norm.fit(error)
norm_sample = stats.norm(mean,std).rvs(len(error))

ax = plt.subplot(311)
sns.distplot(error, bins=100, norm_hist=True)
plt.subplot(312, sharex=ax)
sns.distplot(skew_sample, bins=100, norm_hist=True)
plt.subplot(313, sharex=ax)
sns.distplot(norm_sample, bins=100, norm_hist=True)

plt.show()
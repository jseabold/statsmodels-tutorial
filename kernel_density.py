# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Kernel Density Estimation

# <rawcell>

# A quick univariate example. One of our Google Summer of Code students is working on Multivariate KDE, Non-parametric regression, and extensions.

# <codecell>

np.random.seed(12345)
from statsmodels.distributions.mixture_rvs import mixture_rvs

# <codecell>

obs_dist1 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))

# <codecell>

kde = sm.nonparametric.KDE(obs_dist1)
kde.fit()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist1, bins=50, normed=True, color='red')
ax.plot(kde.support, kde.density, lw=2, color='black');

# <codecell>

obs_dist2 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.beta],
            kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=1,args=(1,.5))))

kde2 = sm.nonparametric.KDE(obs_dist2)
kde2.fit()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist2, bins=50, normed=True, color='red')
ax.plot(kde2.support, kde2.density, lw=2, color='black');
# I think the y tick values are wrong?

# <rawcell>

# The fitted KDE object is a full non-parametric distribution.

# <codecell>

obs_dist3 = mixture_rvs([.25,.75], size=1000, dist=[stats.norm, stats.norm],
                kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
kde3 = sm.nonparametric.KDE(obs_dist3)
kde3.fit()

# <codecell>

kde3.entropy

# <codecell>

kde3.evaluate(-1)

# <headingcell level=4>

# CDF

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.cdf);

# <headingcell level=4>

# Cumulative Hazard Function

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.cumhazard);

# <headingcell level=4>

# Inverse CDF

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.icdf);

# <headingcell level=4>

# Survival Function

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(kde3.support, kde3.sf);


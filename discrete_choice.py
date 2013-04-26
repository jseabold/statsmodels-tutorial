# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Discrete Choice Models - Fair's Affair data

# <markdowncell>

# A survey of women only was conducted in 1974 by *Redbook* asking about extramarital affairs.

# <codecell>

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols

# <codecell>

print sm.datasets.fair.SOURCE

# <codecell>

print sm.datasets.fair.NOTE

# <codecell>

dta = sm.datasets.fair.load_pandas().data

# <codecell>

dta['affair'] = (dta['affairs'] > 0).astype(float)
print dta.head(10)

# <codecell>

print dta.describe()

# <codecell>

affair_mod = logit("affair ~ occupation + educ + occupation_husb" 
                   "+ rate_marriage + age + yrs_married + children"
                   " + religious", dta).fit()

# <codecell>

print affair_mod.summary()

# <rawcell>

# How well are we predicting?

# <codecell>

affair_mod.pred_table()

# <rawcell>

# The coefficients of the discrete choice model do not tell us much. What we're after is marginal effects.

# <codecell>

mfx = affair_mod.get_margeff()
print mfx.summary()

# <codecell>

respondent1000 = dta.ix[1000]
print respondent1000

# <codecell>

resp = dict(zip(range(1,9), respondent1000[["occupation", "educ", 
                                            "occupation_husb", "rate_marriage", 
                                            "age", "yrs_married", "children", 
                                            "religious"]].tolist()))
resp.update({0 : 1})
print resp

# <codecell>

mfx = affair_mod.get_margeff(atexog=resp)
print mfx.summary()

# <codecell>

affair_mod.predict(respondent1000)

# <codecell>

affair_mod.fittedvalues[1000]

# <codecell>

affair_mod.model.cdf(affair_mod.fittedvalues[1000])

# <rawcell>

# The "correct" model here is likely the Tobit model. We have an work in progress branch "tobit-model" on github, if anyone is interested in censored regression models.

# <headingcell level=3>

# Exercise: Logit vs Probit

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.cdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.cdf(support), label='Probit')
ax.legend();

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
support = np.linspace(-6, 6, 1000)
ax.plot(support, stats.logistic.pdf(support), 'r-', label='Logistic')
ax.plot(support, stats.norm.pdf(support), label='Probit')
ax.legend();

# <rawcell>

# Compare the estimates of the Logit Fair model above to a Probit model. Does the prediction table look better? Much difference in marginal effects?

# <headingcell level=3>

# Genarlized Linear Model Example

# <codecell>

print sm.datasets.star98.SOURCE

# <codecell>

print sm.datasets.star98.DESCRLONG

# <codecell>

print sm.datasets.star98.NOTE

# <codecell>

dta = sm.datasets.star98.load_pandas().data
print dta.columns

# <codecell>

print dta[['NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP', 'PERMINTE']].head(10)

# <codecell>

print dta[['AVYRSEXP', 'AVSALK', 'PERSPENK', 'PTRATIO', 'PCTAF', 'PCTCHRT', 'PCTYRRND']].head(10)

# <codecell>

formula = 'NABOVE + NBELOW ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT '
formula += '+ PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'

# <headingcell level=4>

# Aside: Binomial distribution

# <rawcell>

# Toss a six-sided die 5 times, what's the probability of exactly 2 fours?

# <codecell>

stats.binom(5, 1./6).pmf(2)

# <codecell>

from scipy.misc import comb
comb(5,2) * (1/6.)**2 * (5/6.)**3

# <codecell>

from statsmodels.formula.api import glm
glm_mod = glm(formula, dta, family=sm.families.Binomial()).fit()

# <codecell>

print glm_mod.summary()

# <rawcell>

# The number of trials 

# <codecell>

glm_mod.model.data.orig_endog.sum(1)

# <codecell>

glm_mod.fittedvalues * glm_mod.model.data.orig_endog.sum(1)

# <rawcell>

# First differences: We hold all explanatory variables constant at their means and manipulate the percentage of low income households to assess its impact
# on the response variables:

# <codecell>

exog = glm_mod.model.data.orig_exog # get the dataframe

# <codecell>

means25 = exog.mean()
print means25

# <codecell>

means25['LOWINC'] = exog['LOWINC'].quantile(.25)
print means25

# <codecell>

means75 = exog.mean()
means75['LOWINC'] = exog['LOWINC'].quantile(.75)
print means75

# <codecell>

resp25 = glm_mod.predict(means25)
resp75 = glm_mod.predict(means75)
diff = resp75 - resp25

# <rawcell>

# The interquartile first difference for the percentage of low income households in a school district is:

# <codecell>

print "%2.4f%%" % (diff[0]*100)

# <codecell>

nobs = glm_mod.nobs
y = glm_mod.model.endog
yhat = glm_mod.mu

# <codecell>

from statsmodels.graphics.api import abline_plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, ylabel='Observed Values', xlabel='Fitted Values')
ax.scatter(yhat, y)
y_vs_yhat = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
fig = abline_plot(model_results=y_vs_yhat, ax=ax)

# <headingcell level=4>

# Plot fitted values vs Pearson residuals

# <markdowncell>

# Pearson residuals are defined to be 
# 
# $$\frac{(y - \mu)}{\sqrt{(var(\mu))}}$$
# 
# where var is typically determined by the family. E.g., binomial variance is $np(1 - p)$

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, title='Residual Dependence Plot', xlabel='Fitted Values',
                          ylabel='Pearson Residuals')
ax.scatter(yhat, stats.zscore(glm_mod.resid_pearson))
ax.axis('tight')
ax.plot([0.0, 1.0],[0.0, 0.0], 'k-');

# <headingcell level=4>

# Histogram of standardized deviance residuals with Kernel Density Estimate overlayed

# <markdowncell>

# The definition of the deviance residuals depends on the family. For the Binomial distribution this is 
# 
# $$r_{dev} = sign\(Y-\mu\)*\sqrt{2n(Y\log\frac{Y}{\mu}+(1-Y)\log\frac{(1-Y)}{(1-\mu)}}$$
# 
# They can be used to detect ill-fitting covariates

# <codecell>

resid = glm_mod.resid_deviance
resid_std = stats.zscore(resid) 
kde_resid = sm.nonparametric.KDEUnivariate(resid_std)
kde_resid.fit()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, title="Standardized Deviance Residuals")
ax.hist(resid_std, bins=25, normed=True);
ax.plot(kde_resid.support, kde_resid.density, 'r');

# <headingcell level=4>

# QQ-plot of deviance residuals

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = sm.graphics.qqplot(resid, line='r', ax=ax)


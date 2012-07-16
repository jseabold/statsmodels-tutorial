# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Discrete Choice Models - Fair's Affair data

# <rawcell>

# A survey of women only was conducted in 1974 by Redbook asking about extramarital affairs.

# <codecell>

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

mfx = affair_mod.margeff()
print pandas.Series(mfx, index=affair_mod.params.index[1:])

# <codecell>

respondent1000 = dta.ix[1000]
print respondent1000

# <codecell>

resp = dict(zip(range(1,9), respondent1000[["occupation", "educ", "occupation_husb", "rate_marriage", "age", "yrs_married", "children", "religious"]].tolist()))
resp.update({0 : 1})
print resp

# <codecell>

mfx = affair_mod.margeff(atexog=resp)
print pandas.Series(mfx, index=affair_mod.params.index[1:])

# <rawcell>

# We do have a problem in that we have used an inefficient estimator, and our coefficients are biased. This can lead to large differences in the estimated marginal effects. The "correct" model here is likely the Tobit model. We have an work in progress branch "tobit-model" on github, if anyone is interested in censored regression models.

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

# Compare the estimates of the Logit Fair model above to a Probit model.

# <headingcell level=3>

# GLM Example


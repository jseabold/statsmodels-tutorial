# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# This notebook introduces the use of pandas and the formula framework in statsmodels in the context of linear modeling.

# <headingcell level=4>

# It is based heavily on Jonathan Taylor's class notes that use R
# http://www.stanford.edu/class/stats191/interactions.html

# <headingcell level=3>

# Formula interface may change slightly before 0.5.0 release.

# <codecell>

from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot, qqplot
from statsmodels.stats.api import anova_lm

# <headingcell level=2>

# Example 1: IT salary data

# <rawcell>

# Outcome:    S, salaries for IT staff in a corporation
# Predictors: X, experience in years
#             M, managment, 2 levels, 0=non-management, 1=management
#             E, education, 3 levels, 1=Bachelor's, 2=Master's, 3=Ph.D

# <codecell>

try:
    salary_table = pandas.read_csv('salary.table')
except:
    url = 'http://stats191.stanford.edu/data/salary.table'
    salary_table = pandas.read_table(url) # needs pandas 0.7.3
    salary_table.to_csv('salary.table', index=False)

# <codecell>

print salary_table.head(10)

# <codecell>

E = salary_table.E # Education
M = salary_table.M # Management
X = salary_table.X # Experience
S = salary_table.S # Salary

# <rawcell>

# A context manager that mimics R's with() function can be found here: https://gist.github.com/2347382

# <rawcell>

# Let's explore the data

# <codecell>

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, xlabel='Experience', ylabel='Salary',
            xlim=(0, 27), ylim=(9600, 28800))
symbols = ['D', '^']
man_label = ["Non-Mgmt", "Mgmt"]
educ_label = ["Bachelors", "Masters", "PhD"]
colors = ['r', 'g', 'blue']
factor_groups = salary_table.groupby(['E','M'])
for values, group in factor_groups:
    i,j = values
    label = "%s - %s" % (man_label[j], educ_label[i-1])
    ax.scatter(group['X'], group['S'], marker=symbols[j], color=colors[i-1],
               s=350, label=label)
ax.legend(scatterpoints=1, markerscale=.7, labelspacing=1);

# <markdowncell>

# Fit a linear model
# 
# $$S_i = \beta_0 + \beta_1X_i + \beta_2E_{i2} + \beta_3E_{i3} + \beta_4M_i + \epsilon_i$$
# 
# where
# 
# $$ E_{i2}=\cases{1,&if $E_i=2$;\cr 0,&otherwise. \cr}$$ 
# $$ E_{i3}=\cases{1,&if $E_i=3$;\cr 0,&otherwise. \cr}$$ 

# <codecell>

formula = 'S ~ C(E) + C(M) + X'
lm = ols(formula, salary_table).fit()
print lm.summary()

# <headingcell level=2>

# Aside: Contrasts (see contrasts notebook)

# <rawcell>

# Look at the design matrix created for us. Every results instance has a reference to the model.

# <codecell>

lm.model.exog[:10]

# <rawcell>

# Since we initially passed in a DataFrame, we have a transformed DataFrame available.

# <codecell>

print lm.model._data._orig_exog.head(10)

# <rawcell>

# There is a reference to the original untouched data in

# <codecell>

print lm.model._data.frame.head(10)

# <rawcell>

# If you use the formula interface, statsmodels remembers this transformation. Say you want to know the predicted salary for someone with 12 years experience and a Master's degree who is in a management position

# <codecell>

lm.predict({'X' : [12], 'M' : [1], 'E' : [2]})

# <rawcell>

# So far we've assumed that the effect of experience is the same for each level of education and professional role.
# Perhaps this assumption isn't merited. We can formally test this using some interactions.

# <rawcell>

# We can start by seeing if our model assumptions are met. Let's look at a residuals plot.

# <rawcell>

# And some formal tests

# <rawcell>

# Plot the residuals within the groups separately.

# <codecell>

resid = lm.resid

# <codecell>

fig = plt.figure(figsize=(12,8))
xticks = []
ax = fig.add_subplot(111, xlabel='Group (E, M)', ylabel='Residuals')
for values, group in factor_groups:
    i,j = values
    xticks.append(str((i, j)))
    group_num = i*2 + j - 1 # for plotting purposes
    x = [group_num] * len(group)
    ax.scatter(x, resid[group.index], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.set_xticks([1,2,3,4,5,6])
ax.set_xticklabels(xticks)
ax.axis('tight');

# <markdowncell>

# Add an interaction between salary and experience, allowing different intercepts for level of experience.
# 
# $$S_i = \beta_0+\beta_1X_i+\beta_2E_{i2}+\beta_3E_{i3}+\beta_4M_i+\beta_5E_{i2}X_i+\beta_6E_{i3}X_i+\epsilon_i$$

# <codecell>

interX_lm = ols('S ~ C(E)*X + C(M)', salary_table).fit()
print interX_lm.summary()

# <markdowncell>

# Test that $B_5 = \beta_6 = 0$. We can use anova_lm or we can use an F-test.

# <codecell>

print anova_lm(lm, interX_lm)

# <codecell>

print interX_lm.f_test('C(E)[T.2]:X = C(E)[T.3]:X = 0')

# <codecell>

print interX_lm.f_test([[0,0,0,0,0,1,-1],[0,0,0,0,0,0,1]])

# <rawcell>

# The contrasts are created here under the hood by patsy.

# <markdowncell>

# Recall that F-tests are of the form $R\beta = q$

# <codecell>

LC = interX_lm.model._data._orig_exog.design_info.linear_constraint('C(E)[T.2]:X = C(E)[T.3]:X = 0')
print LC.coefs
print LC.constants

# <rawcell>

# Interact education with management

# <codecell>

interM_lm = ols('S ~ X + C(E)*C(M)', salary_table).fit()
print interM_lm.summary()

# <codecell>

print anova_lm(lm, interM_lm)

# <codecell>

infl = interM_lm.get_influence()
resid = infl.resid_studentized_internal

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='X', ylabel='standardized resids')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], resid[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.axis('tight');

# <rawcell>

# There looks to be an outlier.

# <codecell>

outl = interM_lm.outlier_test('fdr_bh')
outl.sort('unadj_p', inplace=True)
print outl

# <codecell>

idx = salary_table.index.drop(32)

# <codecell>

print idx

# <codecell>

lm32 = ols('S ~ C(E) + X + C(M)', df=salary_table, subset=idx).fit()
print lm32.summary()

# <codecell>

interX_lm32 = ols('S ~ C(E) * X + C(M)', df=salary_table, subset=idx).fit()
print interX_lm32.summary()

# <codecell>

table3 = anova_lm(lm32, interX_lm32)
print table3

# <codecell>

interM_lm32 = ols('S ~ X + C(E) * C(M)', df=salary_table, subset=idx).fit()
print anova_lm(lm32, interM_lm32)

# <rawcell>

# Re-plotting the residuals

# <codecell>

resid = interM_lm32.get_influence().summary_frame()['standard_resid']
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='X[~[32]]', ylabel='standardized resids')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], resid[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.axis('tight');

# <rawcell>

# A final plot of the fitted values

# <codecell>

lm_final = ols('S ~ X + C(E)*C(M)', df = salary_table.drop([32])).fit()
mf = lm_final.model._data._orig_exog
lstyle = ['-','--']

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='Experience', ylabel='Salary')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], S[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
    # drop NA because there is no idx 32 in the final model
    ax.plot(mf.X[idx].dropna(), lm_final.fittedvalues[idx].dropna(),
            ls=lstyle[j], color=colors[i-1])
ax.axis('tight');

# <rawcell>

# From our first look at the data, the difference between Master's and PhD in the management group is different than in the non-management group. This is an interaction between the two qualitative variables management, M and education, E. We can visualize this by first removing the effect of experience, then plotting the means within each of the 6 groups using interaction.plot.

# <codecell>

U = S - X * interX_lm32.params['X']
U.name = 'Salary|X'

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = interaction_plot(E, M, U, colors=['red','blue'], markers=['^','D'],
        markersize=10, ax=ax)

# <headingcell level=3>

# Minority Employment Data - ABLine plotting

# <rawcell>

# TEST  - Job Aptitude Test Score
# ETHN  - 1 if minority, 0 otherwise
# JPERF - Job performance evaluation

# <codecell>

try:
    minority_table = pandas.read_table('minority.table')
except: # don't have data already
    url = 'http://stats191.stanford.edu/data/minority.table'
    minority_table = pandas.read_table(url)

# <codecell>

factor_group = minority_table.groupby(['ETHN'])

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
colors = ['purple', 'green']
markers = ['o', 'v']
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)
ax.legend(['ETHN == 1', 'ETHN == 0'], scatterpoints=1)

# <codecell>

min_lm = ols('JPERF ~ TEST', df=minority_table).fit()
print min_lm.summary()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)
ax.legend(['ETHN == 1', 'ETHN == 0'], scatterpoints=1, loc='upper left')
fig = abline_plot(model_results = min_lm, ax=ax)

# <codecell>

min_lm2 = ols('JPERF ~ TEST + TEST:ETHN', df=minority_table).fit()
print min_lm2.summary()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = min_lm2.params['Intercept'],
                 slope = min_lm2.params['TEST'], ax=ax, color='purple')
ax = fig.axes[0]
fig = abline_plot(intercept = min_lm2.params['Intercept'],
        slope = min_lm2.params['TEST'] + min_lm2.params['TEST:ETHN'],
        ax=ax, color='green')
ax.legend(['ETHN == 1', 'ETHN == 0'], scatterpoints=1, loc='upper left');

# <codecell>

min_lm3 = ols('JPERF ~ TEST + ETHN', df = minority_table).fit()
print min_lm3.summary()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = min_lm3.params['Intercept'],
                 slope = min_lm3.params['TEST'], ax=ax, color='purple')

ax = fig.axes[0]
fig = abline_plot(intercept = min_lm3.params['Intercept'] + min_lm3.params['ETHN'],
        slope = min_lm3.params['TEST'], ax=ax, color='green')
ax.legend(['ETHN == 1', 'ETHN == 0'], scatterpoints=1, loc='upper left');

# <codecell>

min_lm4 = ols('JPERF ~ TEST * ETHN', df = minority_table).fit()
print min_lm4.summary()

# <codecell>

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, ylabel='JPERF', xlabel='TEST')
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = min_lm4.params['Intercept'],
                 slope = min_lm4.params['TEST'], ax=ax, color='purple')
ax = fig.axes[0]
fig = abline_plot(intercept = min_lm4.params['Intercept'] + min_lm4.params['ETHN'],
        slope = min_lm4.params['TEST'] + min_lm4.params['TEST:ETHN'],
        ax=ax, color='green')
ax.legend(['ETHN == 1', 'ETHN == 0'], scatterpoints=1, loc='upper left');

# <rawcell>

# Is there any effect of ETHN on slope or intercept?
# Y ~ TEST vs. Y ~ TEST + ETHN + ETHN:TEST

# <codecell>

table5 = anova_lm(min_lm, min_lm4)
print table5

# <rawcell>

# Is there any effect of ETHN on intercept?
# Y ~ TEST vs. Y ~ TEST + ETHN

# <codecell>

table6 = anova_lm(min_lm, min_lm3)
print table6

# <rawcell>

# Is there any effect of ETHN on slope?
# Y ~ TEST vs. Y ~ TEST + ETHN:TEST

# <codecell>

table7 = anova_lm(min_lm, min_lm2)
print table7

# <rawcell>

# Is it just the slope or both?
# Y ~ TEST + ETHN:TEST vs Y ~ TEST + ETHN + ETHN:TEST

# <codecell>

table8 = anova_lm(min_lm2, min_lm4)
print table8

# <headingcell level=3>

# Two Way ANOVA - Kidney failure data

# <rawcell>

# Weight - (1,2,3) - Level of weight gan between treatments
# Duration - (1,2) - Level of duration of treatment
# Days - Time of stay in hospital

# <codecell>

try:
    kidney_table = pandas.read_table('./kidney.table')
except:
    url = 'http://stats191.stanford.edu/data/kidney.table'
    kidney_table = pandas.read_table(url, delimiter=" *")

# <codecell>

# Explore the dataset, it's a balanced design
print kidney_table.groupby(['Weight', 'Duration']).size()

# <codecell>

kt = kidney_table
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
fig = interaction_plot(kt['Weight'], kt['Duration'], np.log(kt['Days']+1),
        colors=['red', 'blue'], markers=['D','^'], ms=10, ax=ax)

# <markdowncell>

# $$Y_{ijk} = \mu + \alpha_i + \beta_j + \(\alpha\beta\)_{ij}+\epsilon_{ijk}$$
# 
# with 
# 
# $$\epsilon_{ijk}\sim N\(0,\sigma^2\)$$

# <codecell>

help(anova_lm)

# <rawcell>

# Things available in the calling namespace are available in the formula evaluation namespace

# <codecell>

kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight)', df=kt).fit()

# <rawcell>

# ANOVA Type-I Sum of Squares
# 
# SS(A) for factor A. 
# SS(B|A) for factor B. 
# SS(AB|B, A) for interaction AB. 

# <codecell>

print anova_lm(kidney_lm)

# <rawcell>

# ANOVA Type-II Sum of Squares
# 
# SS(A|B) for factor A. 
# SS(B|A) for factor B. 

# <codecell>

print anova_lm(kidney_lm, typ=2)

# <rawcell>

# ANOVA Type-III Sum of Squares
# 
# SS(A|B, AB) for factor A. 
# SS(B|A, AB) for factor B. 

# <codecell>

print anova_lm(ols('np.log(Days+1) ~ C(Duration, Sum) * C(Weight, Poly)', df=kt).fit(), typ=3)

# <headingcell level=4>

# Excercise: Find the 'best' model for the kidney failure dataset


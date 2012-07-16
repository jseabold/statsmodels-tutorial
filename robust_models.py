# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# M-Estimators for Robust Linear Modeling

# <markdowncell>

# An M-estimator minimizes the function 
# 
# $$Q(e_i, \rho) = \sum_i\rho(\frac{e_i}{s})$$
# 
# where $\rho$ is a symmetric function of the residuals intended to reduce the influence of outliers and $s$ is an estimate of scale. The robust estimates $\hat{\beta}$ are computed by iteratively re-weighted least squares.

# <rawcell>

# We have several choices available for these weighting functions

# <codecell>

norms = sm.robust.norms

# <codecell>

def plot_weights(support, weights_func, xlabels, xticks):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(support, weights_func(support))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_ylim(-.1, 1.1)
    return ax

# <headingcell level=3>

# Andrew's Wave

# <codecell>

help(norms.AndrewWave.weights)

# <codecell>

a = 1.339
support = np.linspace(-np.pi*a, np.pi*a, 100)
andrew = norms.AndrewWave(a=a)
plot_weights(support, andrew.weights, ['$-\pi*a$', '0', '$\pi*a$'], [-np.pi*a, 0, np.pi*a]);

# <headingcell level=3>

# Hampel's 17A

# <codecell>

c = 8
support = np.linspace(-3*c, 3*c, 1000)
hampel = norms.Hampel(a=2., b=4., c=c)
plot_weights(support, hampel.weights, ['3*c', '0', '3*c'], [-3*c, 0, 3*c]);

# <headingcell level=3>

# Huber's t

# <codecell>

t = 1.345
support = np.linspace(-3*t, 3*t, 1000)
huber = norms.HuberT(t=t)
plot_weights(support, huber.weights, ['-3*t', '0', '3*t'], [-3*t, 0, 3*t]);

# <headingcell level=3>

# Least Squares

# <codecell>

support = np.linspace(-3, 3, 1000)
lst_sq = norms.LeastSquares()
plot_weights(support, lst_sq.weights, ['-3', '0', '3'], [-3, 0, 3]);

# <headingcell level=3>

# Ramsay's Ea

# <codecell>

a = .3
support = np.linspace(-3*a, 3*a, 1000)
ramsay = norms.RamsayE(a=a)
plot_weights(support, ramsay.weights, ['-3*a', '0', '3*a'], [-3*a, 0, 3*a]);

# <headingcell level=3>

# Trimmed Mean

# <codecell>

c = 2
support = np.linspace(-3*c, 3*c, 1000)
trimmed = norms.TrimmedMean(c=c)
plot_weights(support, trimmed.weights, ['-3*c', '0', '3*c'], [-3*c, 0, 3*c]);

# <headingcell level=3>

# Tukey's Biweight

# <codecell>

c = 4.685
support = np.linspace(-3*c, 3*c, 1000)
tukey = norms.TukeyBiweight(c=c)
plot_weights(support, tukey.weights, ['-3*c', '0', '3*c'], [-3*c, 0, 3*c]);

# <headingcell level=3>

# Scale Estimators

# <rawcell>

# The default is MAD - Median Absolute Deviation, but another popular choice is Huber's proposal 2

# <codecell>

np.random.seed(12345)
fat_tails = stats.t(6).rvs(40)

# <codecell>

print fat_tails.mean(), fat_tails.std()

# <codecell>

print stats.norm.fit(fat_tails)

# <codecell>

print stats.t.fit(fat_tails, f0=6)

# <codecell>

huber = sm.robust.scale.Huber()
loc, scale = huber(fat_tails)
print loc, scale

# <codecell>

sm.robust.stand_mad(fat_tails)

# <codecell>

sm.robust.scale.mad(fat_tails)

# <headingcell level=3>

# Duncan's Occupational Prestige data - M-estimation for outliers

# <codecell>

from statsmodels.graphics.api import abline_plot
from pandas.rpy import load_data
from statsmodels.formula.api import ols, rlm

# <codecell>

prestige = load_data('Duncan', 'car')
print prestige.head(10)

# <codecell>

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(211)
ax1.scatter(prestige.income, prestige.prestige)
ax1.set_xlabel('Income')
ax1.set_ylabel('Prestige')
xy_outlier = prestige.ix['minister'][['income','prestige']]
ax1.annotate('Minister', xy_outlier, xy_outlier+1, fontsize=16)
ax2 = fig.add_subplot(212)
ax2.scatter(prestige.education, prestige.prestige)
ax2.set_xlabel('Education')
ax2.set_ylabel('Prestige');

# <codecell>

ols_model = ols('prestige ~ income + education', prestige).fit()
print ols_model.summary()

# <codecell>

infl = ols_model.get_influence()
student = infl.summary_frame()['student_resid']
print student

# <codecell>

print student.ix[np.abs(student) > 2]

# <codecell>

print infl.summary_frame().ix['minister']

# <codecell>

print ols_model.outlier_test('sidak')

# <codecell>

rlm_model = rlm('prestige ~ income + education', prestige).fit()
print rlm_model.summary()

# <codecell>

print rlm_model.weights

# <headingcell level=3>

# Hertzprung Russell data for Star Cluster CYG 0B1 - Leverage Points

# <rawcell>

# Data is on the luminosity and temperature of 47 stars in the direction of Cygnus.

# <codecell>

dta = load_data('starsCYG', package='robustbase')

# <codecell>

from matplotlib.patches import Ellipse
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, xlabel='log(Temp)', ylabel='log(Light)', title='Hertzsprung-Russell Diagram of Star Cluster CYG OB1')
ax.scatter(*dta.values.T)
# highlight outliers
e = Ellipse((3.5, 6), .2, 1, alpha=.25, color='r')
ax.add_patch(e);
ax.annotate('Red giants', xy=(3.6, 6), xytext=(3.8, 6),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            horizontalalignment='left', verticalalignment='bottom',
            clip_on=True, # clip to the axes bounding box
            fontsize=16,
     )
# annotate these with their index
for i,row in dta.ix[dta['log.Te'] < 3.8].iterrows():
    ax.annotate(i, row, row + .01, fontsize=14)
xlim, ylim = ax.get_xlim(), ax.get_ylim()

# <codecell>

from IPython.display import Image
Image(filename='star_diagram.png')

# <codecell>

y = dta['log.light']
X = sm.add_constant(dta['log.Te'], prepend=True)
ols_model = sm.OLS(y, X).fit()
abline_plot(model_results=ols_model, ax=ax)

# <codecell>

rlm_mod = sm.RLM(y, X, sm.robust.norms.TrimmedMean(.5)).fit()
abline_plot(model_results=rlm_mod, ax=ax, color='red')

# <rawcell>

# Why? Because M-estimators are not robust to leverage points.

# <codecell>

infl = ols_model.get_influence()
h_bar = 2*(ols_model.df_model + 1 )/ols_model.nobs
hat_diag = infl.summary_frame()['hat_diag']
hat_diag.ix[hat_diag > h_bar]

# <codecell>

print ols_model.outlier_test('sidak')

# <rawcell>

# Let's delete that line

# <codecell>

del ax.lines[-1]

# <codecell>

weights = np.ones(len(X))
weights[X[X['log.Te'] < 3.8].index.values - 1] = 0
wls_model = sm.WLS(y, X, weights=weights).fit()
abline_plot(model_results=wls_model, ax=ax, color='green')

# <rawcell>

# MM estimators are good for this type of problem, unfortunately, we don't yet have these yet. It's being worked on by one of our Google Summer of Code students. We have a good excuse to look at the R cell magics in the notebook.

# <codecell>

yy = y.values[:,None]
xx = X['log.Te'].values[:,None]

# <codecell>

%load_ext rmagic

%Rpush yy xx
%R mod <- lmrob(yy ~ xx);
%R params <- mod$coefficients;
%Rpull params

# <codecell>

%R print(mod)

# <codecell>

print params

# <codecell>

abline_plot(intercept=params[0], slope=params[1], ax=ax, color='green')


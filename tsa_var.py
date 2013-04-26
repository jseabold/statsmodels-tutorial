# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Vector Autoregressions: inflation-unemployment-interest rate

# <rawcell>

# Vector Autoregression (VAR), introduced by Nobel laureate Christopher Sims in 1980, is a powerful statistical tool in the macroeconomist's toolkit.

# <markdowncell>

# Formally a VAR model is
# 
# $$Y_t = A_1 Y_{t-1} + \ldots + A_p Y_{t-p} + u_t$$
# 
# $$u_t \sim {\sf Normal}(0, \Sigma_u)$$
# 
# where $Y_t$ is of dimension $K$ and $A_i$ is a $K \times K$ coefficient matrix.

# <codecell>

dta = sm.datasets.macrodata.load_pandas().data
endog = dta[["infl", "unemp", "tbilrate"]]

# <codecell>

index = sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3')
dta.index = pandas.Index(index)
del dta['year']
del dta['quarter']
endog.index = pandas.Index(index) # DatetimeIndex or PeriodIndex in 0.8.0
print endog.head(10)

# <codecell>

endog.plot(subplots=True, figsize=(14,18));

# <codecell>

# model only after Volcker appointment
var_mod = sm.tsa.VAR(endog.ix['1979-12-31':]).fit(maxlags=4, ic=None)
print var_mod.summary()

# <headingcell level=3>

# Diagnostics

# <codecell>

np.abs(var_mod.roots)

# <rawcell>

# var_mod.test_normality() and var_mod.test_whiteness() are also available. There are problems with this model...

# <headingcell level=3>

# Granger-Causality tests

# <codecell>

var_mod.test_causality('unemp', 'tbilrate', kind='Wald')

# <codecell>

table = pandas.DataFrame(np.zeros((9,3)), columns=['chi2', 'df', 'prob(>chi2)'])
index = []
variables = set(endog.columns.tolist())
i = 0
for vari in variables:
    others = []
    for j,ex_vari in enumerate(variables):
        if vari == ex_vari: # don't want to test this
            continue
        others.append(ex_vari)
        res = var_mod.test_causality(vari, ex_vari, kind='Wald', verbose=False)
        table.ix[[i], ['chi2', 'df', 'prob(>chi2)']] = (res['statistic'], res['df'], res['pvalue'])
        i += 1
        index.append([vari, ex_vari])
    res = var_mod.test_causality(vari, others, kind='Wald', verbose=False)
    table.ix[[i], ['chi2', 'df', 'prob(>chi2)']] = res['statistic'], res['df'], res['pvalue']
    index.append([vari, 'ALL'])
    i += 1
table.index = pandas.MultiIndex.from_tuples(index, names=['Equation', 'Excluded'])

# <codecell>

print table

# <markdowncell>

# From this we reject the null that these variables do not Granger cause for all cases except for infl -> tbilrate. In other words, in almost all cases we can reject the null hypothesis that the lags of the *excluded* variable are jointly zero in *Equation*.

# <headingcell level=3>

# Order Selection

# <codecell>

var_mod.model.select_order()

# <headingcell level=3>

# Impulse Response Functions

# <rawcell>

# Suppose we want to examine what happens to each of the variables when a 1 unit increase in the current value of one of the VAR errors occurs (a "shock"). To isolate the effects of only one error while holding the others constant, we need the model to be in a form so that the contemporaneous errors are uncorrelated across equations. One such way to achieve this is the so-called recursive VAR. In the recursive VAR, the order of the variables is determined by how the econometrician views the economic processes as ocurring. Given this order, inflation is determined by the contemporaneous unemployment rate and tbilrate is determined by the contemporaneous inflation and unemployment rates. Unemployment is a function of only the past values of itself, inflation, and the T-bill rate.
# 
# We achieve such a structure by using the Choleski decomposition.

# <codecell>

irf = var_mod.irf(24)

# <codecell>

irf.plot(orth=True, signif=.33, subplot_params = {'fontsize' : 18})

# <rawcell>

# Note that inflation dynamics are not very persistent, but do appear to have a significant and immediate impact on interest rates and on unemployment in the medium run.

# <headingcell level=3>

# Forecast Error Decompositions

# <codecell>

var_mod.fevd(24).summary()

# <codecell>

var_mod.fevd(24).plot(figsize=(12,12))

# <rawcell>

# There is some amount of interaction between the variables. For instance, at the 12 quarter horizon, 40% of the error in the forecast of the T-bill rate is attributed to the inflation and unemployment shocks in the recursive VAR.

# <rawcell>

# To make structural inferences - e.g., what is the effect on the rate of inflation and unemployment of an unexpected 100 basis point increase in the Federal Funds rate (proxied by the T-bill rate here), we might want to fit a structural VAR model based on economic theory of monetary policy. For instance, we might replace the VAR equation for the T-bill rate with a policy equation such as a Taylor rule and restrict coefficients. You can do so with the sm.tsa.SVAR class.

# <headingcell level=3>

# Exercises

# <markdowncell>

# Experiment with different VAR models. You can try to adjust the number of lags in the VAR model calculated above or the ordering of the variables and see how it affects the model.

# <markdowncell>

# You might also try adding variables to the VAR, say *M1* measure of money supply, or estimating a different model using measures of consumption (realcons), government spending (realgovt), or GDP (realgdp).


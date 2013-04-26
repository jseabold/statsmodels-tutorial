# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

# <codecell>

print sm.datasets.spector.NOTE

# <codecell>

data = sm.datasets.spector.load_pandas()
exog = sm.add_constant(data.exog, prepend=True)
endog = data.endog

# <codecell>

sm_probit = sm.Probit(endog, exog).fit()

# <rawcell>

# * To create your own Likelihood Model, you just need to overwrite the loglike method.

# <codecell>

class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q*np.dot(exog, params)).sum()

# <codecell>

my_probit = MyProbit(endog, exog).fit()

# <codecell>

print sm_probit.params

# <codecell>

print sm_probit.cov_params()

# <codecell>

print my_probit.params

# <rawcell>

# You can get the variance-covariance of the parameters. Notice that we didn't have to provide Hessian or Score functions.

# <codecell>

print my_probit.cov_params()


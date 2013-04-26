# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Learn More and Get Help

# <markdowncell>

# Documentation: http://statsmodels.sf.net
# 
# Mailing List: http://groups.google.com/group/pystatsmodels
# 
# Use the source: https://github.com/statsmodels/statsmodels

# <headingcell level=3>

# Tutorial Import Assumptions

# <codecell>

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
pandas.set_printoptions(notebook_repr_html=False,
                        precision=4,
                        max_columns=12)

# <headingcell level=3>

# Statsmodels Import Convention

# <codecell>

import statsmodels.api as sm

# <markdowncell>

# Import convention for models for which a formula is available.

# <codecell>

from statsmodels.formula.api import ols, rlm, glm, #etc.

# <headingcell level=3>

# Package Overview

# <markdowncell>

# Regression models in statsmodels.regression

# <markdowncell>

# Discrete choice models in statsmodels.discrete

# <markdowncell>

# Robust linear models in statsmodels.robust

# <markdowncell>

# Generalized linear models in statsmodels.genmod

# <markdowncell>

# Time Series Analysis in statsmodels.tsa

# <markdowncell>

# Nonparametric models in statsmodels.nonparametric

# <markdowncell>

# Plotting functions in statsmodels.graphics

# <markdowncell>

# Input/Output in statsmodels.iolib (Foreign data, ascii, HTML, $\LaTeX$ tables)

# <markdowncell>

# Statistical tests, ANOVA in statsmodels.stats

# <markdowncell>

# Datasets in statsmodels.datasets (See also the new GPL package Rdatasets: https://github.com/vincentarelbundock/Rdatasets)

# <headingcell level=3>

# Base Classes

# <codecell>

from statsmodels.base import model

# <codecell>

help(model.Model)

# <codecell>

help(model.LikelihoodModel)

# <codecell>

help(model.LikelihoodModelResults)

# <codecell>

from statsmodels.regression.linear_model import RegressionResults

# <codecell>

help(RegressionResults)


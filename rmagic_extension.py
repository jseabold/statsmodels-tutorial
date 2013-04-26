# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Rmagic Functions Extension

# <codecell>

%pylab inline

# <headingcell level=2>

# Line magics

# <markdowncell>

# * IPython has an `rmagic` extension that contains a some magic functions for working with R via rpy2. 
# * This extension can be loaded using the `%load_ext` magic as follows:

# <codecell>

%load_ext rmagic 

# <markdowncell>

# * We can go from numpy arrays to compute some statistics in R and back
# * Let's suppose we just want to fit a simple linear model to a scatterplot.

# <codecell>

import numpy as np
import pylab
X = np.array([0,1,2,3,4])
Y = np.array([3,5,4,6,7])
pylab.scatter(X, Y)

# <markdowncell>

# * We can accomplish this by first pushing variables to R
# * Then fitting a model
# * And finally returning the results
# * The line magic `%Rpush` copies its arguments to variables of the same name in rpy2
# * The `%R` line magic evaluates the string in rpy2 and returns the result

# <codecell>

%Rpush X Y
%R lm(Y~X)$coef

# <markdowncell>

# We can check that this is correct fairly easily:

# <codecell>

Xr = X - X.mean(); Yr = Y - Y.mean()
slope = (Xr*Yr).sum() / (Xr**2).sum()
intercept = Y.mean() - X.mean() * slope
(intercept, slope)

# <markdowncell>

# It is also possible to return more than one value with %R.

# <codecell>

%R resid(lm(Y~X)); coef(lm(Y~X))

# <markdowncell>

# * One can also easily capture the results of %R into python objects. 
# * Like R, the return value of this multiline expression is the final value, which is the `coef(lm(X~Y))`. 
# * To pull other variables from R, there is one more set of magic functions

# <markdowncell>

# * `%Rpull` and `%Rget` 
# * Both are useful to retrieve variables in the rpy2 namespace
# * The main difference is that one returns the value (%Rget), while the other pulls it to the user's namespace.
# * Imagine we've stored the results of some calculation in the variable `a` in rpy2's namespace. 
# * By using the %R magic, we can obtain these results and store them in b. 
# * We can also pull them directly to the namespace with %Rpull. 
# * Note that they are both views on the same data.

# <codecell>

b = %R a=resid(lm(Y~X))
%Rpull a
print a
assert id(b.data) == id(a.data)
%R -o a

# <markdowncell>

# %Rpull is equivalent to calling %R with just -o

# <codecell>

%R d=resid(lm(Y~X)); e=coef(lm(Y~X))
%R -o d -o e
%Rpull e
print d
print e
import numpy as np
np.testing.assert_almost_equal(d, a)

# <markdowncell>

# On the other hand %Rpush is equivalent to calling %R with just -i and no trailing code.

# <codecell>

A = np.arange(20)
%R -i A
%R mean(A)

# <markdowncell>

# The magic %Rget retrieves one variable from R.

# <codecell>

%Rget A

# <headingcell level=2>

# Plotting and capturing output

# <markdowncell>

# * R's console (i.e. its stdout() connection) is captured by ipython,
# * So are any plots which are published as PNG files
# * As a call to %R may produce a return value (see above), what happens to a magic like the one below?
# * The R code specifies that something is published to the notebook. 
# * If anything is published to the notebook, that call to %R returns None.

# <codecell>

v1 = %R plot(X,Y); print(summary(lm(Y~X))); vv=mean(X)*mean(Y)
print 'v1 is:', v1
v2 = %R mean(X)*mean(Y)
print 'v2 is:', v2

# <headingcell level=2>

# What value is returned from %R?

# <markdowncell>

# * Some calls have no particularly interesting return value, the magic `%R` will not return anything in this case. 
# * The return value in rpy2 is actually NULL so `%R` returns None.

# <codecell>

v = %R plot(X,Y)
assert v == None

# <markdowncell>

# Also, if the return value of a call to %R (inline mode) has just been printed to the console, then its value is also not returned.

# <codecell>

v = %R print(X)
assert v == None

# <markdowncell>

# * If the last value did not print anything to console, the value is returned

# <codecell>

v = %R print(summary(X)); X
print 'v:', v

# <markdowncell>

# * The return value can be suppressed by a trailing ';' or an -n argument

# <codecell>

%R -n X

# <codecell>

%R X; 

# <headingcell level=2>

# Cell level magic

# <markdowncell>

# * What if we want to run several lines of R code
# * This is the cell-level magic.
# * For the cell level magic, inputs can be passed via the -i or --inputs argument in the line
# * These variables are copied from the shell namespace to R's namespace using `rpy2.robjects.r.assign` 
# * It would be nice not to have to copy these into R: rnumpy ( http://bitbucket.org/njs/rnumpy/wiki/API ) has done some work to limit or at least make transparent the number of copies of an array. 
# * Arrays can be output from R via the -o or --outputs argument in the line. All other arguments are sent to R's png function, which is the graphics device used to create the plots.
# * We can redo the above calculations in one ipython cell. 
# * We might also want to add a summary or perhaps the standard plotting diagnostics of the lm.

# <codecell>

%%R -i X,Y -o XYcoef
XYlm = lm(Y~X)
XYcoef = coef(XYlm)
print(summary(XYlm))
par(mfrow=c(2,2))
plot(XYlm)

# <headingcell level=2>

# Passing data back and forth

# <markdowncell>

# * Currently (Summer 2012), data is passed through `RMagics.pyconverter` when going from python to R and `RMagics.Rconverter` when going from R to python. 
# * These currently default to numpy.ndarray. Future work will involve writing better converters, most likely involving integration with http://pandas.sourceforge.net.
# * Passing ndarrays into R requires a copy, though once an object is returned to python, this object is NOT copied, and it is possible to change its values.

# <codecell>

seq1 = np.arange(10)

# <codecell>

%%R -i seq1 -o seq2
seq2 = rep(seq1, 2)
print(seq2)

# <codecell>

seq2[::2] = 0
seq2

# <codecell>

%%R
print(seq2)

# <markdowncell>

# * Once the array data has been passed to R, modifring its contents does not modify R's copy of the data.

# <codecell>

seq1[0] = 200
%R print(seq1)

# <markdowncell>

# * If we pass data as both input and output, then the value of "data" in the user's namespace will be overwritten 
# * the new array will be a view of the data in R's copy.

# <codecell>

print seq1
%R -i seq1 -o seq1
print seq1
seq1[0] = 200
%R print(seq1)
seq1_view = %R seq1
assert(id(seq1_view.data) == id(seq1.data))

# <headingcell level=2>

# Exception handling

# <markdowncell>

# Exceptions are handled by passing back rpy2's exception and the line that triggered it.

# <codecell>

try:
    %R -n nosuchvar
except Exception as e:
    print e

# <headingcell level=2>

# Structured arrays and data frames

# <markdowncell>

# * In R, data frames play an important role as they allow array-like objects of mixed type with column names (and row names). 
# * In numpy, the closest analogy is a structured array with named fields. 
# * In future work, it would be nice to use pandas to return full-fledged DataFrames from rpy2. 
# * In the mean time, structured arrays can be passed back and forth with the -d flag to %R, %Rpull, and %Rget

# <codecell>

datapy= np.array([(1, 2.9, 'a'), (2, 3.5, 'b'), (3, 2.1, 'c')],
          dtype=[('x', '<i4'), ('y', '<f8'), ('z', '|S1')])

# <codecell>

%%R -i datapy -d datar
datar = datapy

# <codecell>

datar

# <codecell>

%R datar2 = datapy
%Rpull -d datar2
datar2

# <codecell>

%Rget -d datar2

# <markdowncell>

# For arrays without names, the -d argument has no effect because the R object has no colnames or names.

# <codecell>

Z = np.arange(6)
%R -i Z
%Rget -d Z

# <markdowncell>

# * For mixed-type data frames in R, if the -d flag is not used, then an array of a single type is returned 
# * Its value is transposed. 
# * This would be nice to fix, but it seems something that should be fixed at the rpy2 level. See [here](https://bitbucket.org/lgautier/rpy2/issue/44/numpyrecarray-as-dataframe)

# <codecell>

%Rget datar2


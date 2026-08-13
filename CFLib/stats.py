from math import *
import numpy as np 
from scipy.stats import kurtosis
from scipy.stats import skew

def stats(x, axis=None, full=False):

    '''
    Given the constant array 'x', stats will return the tuple
    ( E[x], StDev(x) := sqrt( E[ (x - E[x])^2 ] ))
    where E, represents the sample average.
    '''
    if full:
        return x.mean(axis=axis), x.std(axis=axis), skew(x), kurtosis(x)
    else:
        return x.mean(axis=axis), x.std(axis=axis)

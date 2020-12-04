#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:28:51 2020

@author: cd902

Simulate synthetic data

"""

from __future__ import print_function, division
import os
import numpy as np

import misc
import mMR
from stochastic_primal_dual_hybrid_gradient import pdhg, spdhg

import odl
from odl.contrib import fom
from odl.solvers import CallbackPrintIteration, CallbackPrintTiming
from scipy.ndimage.filters import gaussian_filter

# %%


RT = mMR.operator_mmr()
X = RT.domain
Y = RT.range

phantom = odl.phantom.transmission.shepp_logan(X)
# attenuation
support = X.element(phantom.ufuncs.greater(0))    
factors = -RT(0.005 / X.cell_sides[0] * support)
factors.ufuncs.exp(out=factors)
counts_observed = (factors * RT(phantom)).ufuncs.sum()
print(counts_observed)
counts_desired = 5e+6
counts_background = 1e+6
factors *= counts_desired / counts_observed
# background
sino = RT(phantom)
sino_supp = sino.ufuncs.greater(0)
# smooth_supp = RT.range.element(
#     gaussian_filter(sino_supp, sigma=[1, 2 / reco_space.cell_sides[0]]))
smooth_supp = Y.element(
    gaussian_filter(sino_supp, sigma=[1, 1]))
background = 10 * smooth_supp + 10
background *= counts_background / background.ufuncs.sum()








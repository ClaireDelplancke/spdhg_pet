#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:31:07 2020

@author: cd902

Compute target for map tv problem on real and synthetic data
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
import matplotlib.pyplot as plt

# %% REAL DATA

folder_data = '/home/cd902/siemens-biograph_data/amyloidPET_FBP_TP0'
folder_out = '/u/s/cd902/GitHub/spdhg_pet/results/amyloidPET_FBP_TP0'

#%% set parameters and create folder structure
filename = 'map_tv'
dataset = 'amyloid'

nepoch_target = 5000
rho = 0.999
tol_step = 1e-6

folder_norms = '{}/norms'.format(folder_out)
misc.mkdir(folder_norms)

planes = None
alpha = 5
clim = [0, 1]  # colour limits for plots
data_suffix = 'rings0-64_span1_time3000-3600'

def save_image(x, n, f):
    misc.save_image(x.asarray(), n, f, planes=planes, clim=clim)

folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
misc.mkdir(folder_main)
misc.mkdir('{}/py'.format(folder_main))
misc.mkdir('{}/logs'.format(folder_main))

# load real data and convert to odl
file_data = '{}/data_{}.npy'.format(folder_data, data_suffix)
#XXX (data, background, factors, image, image_mr,image_ct) = np.load(file_data, allow_pickle = True)
(data, background, factors, image, image_ct) = np.load(file_data, allow_pickle = True)
Y = mMR.operator_mmr().range
data = Y.element(data)
background = Y.element(background)
factors = Y.element(factors)

# %%

# define operator
K = mMR.operator_mmr(factors=factors)
X = K.domain
norm_K = misc.norm(K, '{}/norm_1subset.npy'.format(folder_norms))
KL = misc.kullback_leibler(Y, data, background)
folder_param = '{}/alpha{:.2g}'.format(folder_main, alpha)
misc.mkdir(folder_param)
misc.mkdir('{}/pics'.format(folder_param))

D = odl.Gradient(X)
norm_D = misc.norm(D, '{}/norm_D.npy'.format(folder_param))

c = norm_K / norm_D
D = odl.Gradient(X) * c
norm_D *= c
L1 = (alpha / c) * odl.solvers.GroupL1Norm(D.range)
L164 = (alpha / c) * odl.solvers.GroupL1Norm(D.range.astype('float64'))
g = odl.solvers.IndicatorBox(X, lower=0)

obj_fun = KL * K + L1 * D + g  # objective functional
 
if not os.path.exists('{}/pics/gray_image_pet.png'
                      .format(folder_param)):
    tmp = X.element()
    tmp_op = mMR.operator_mmr()
    tmp_op.toodl(image, tmp)
    fldr = '{}/pics'.format(folder_param)
    misc.save_image(tmp.asarray(), 'image_pet', fldr, planes=planes)
    #XXX tmp_op.toodl(image_mr, tmp)
    # misc.save_image(tmp.asarray(), 'image_mr', fldr, planes=planes)
    tmp_op.toodl(image_ct, tmp)
    misc.save_image(tmp.asarray(), 'image_ct', fldr, planes=planes)
    
# %%
nepoch_target = 5000

# --- get target --- BE CAREFUL, THIS TAKES TIME
file_target = '{}/target.npy'.format(folder_param)

if not os.path.exists(file_target):
    print('file {} does not exist. Compute it.'.format(file_target))

    A = odl.BroadcastOperator(K, D)
    f = odl.solvers.SeparableSum(KL, L1)

    norm_A = misc.norm(A, '{}/norm_tv.npy'.format(folder_main))
    sigma = rho / norm_A
    tau = rho / norm_A

    niter_target = nepoch_target

    step = 10
    cb = (CallbackPrintIteration(step=step, end=', ') &
          CallbackPrintTiming(step=step, cumulative=False, end=', ') &
          CallbackPrintTiming(step=step, cumulative=True,
                              fmt='total={:.3f} s'))

    x_opt = X.zero()
    odl.solvers.pdhg(x_opt, g, f, A, niter_target, tau, sigma,
                     callback=cb)

    obj_opt = obj_fun(x_opt)

    save_image(x_opt, 'target', '{}/pics'.format(folder_param))
    np.save(file_target, (x_opt, obj_opt))
else:
    print('file {} exists. Load it.'.format(file_target))
    x_opt, obj_opt = np.load(file_target, allow_pickle = True)
    
    

    
# %% SYNTHETIC DATA

# # without hardware attenuation
# folder_data = '/home/cd902/siemens-biograph_data/synthetic_wh'
# folder_out = '/u/s/cd902/GitHub/spdhg_pet/results/synthetic_wh'
    
# with hardware attenuation
folder_data = '/home/cd902/siemens-biograph_data/synthetic'
folder_out = '/u/s/cd902/GitHub/spdhg_pet/results/synthetic'
    
#%% set parameters and create folder structure

filename = 'map_tv'
dataset = 'synthetic'


folder_norms = '{}/norms'.format(folder_out)
misc.mkdir(folder_norms)
folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
misc.mkdir(folder_main)
misc.mkdir('{}/py'.format(folder_main))
misc.mkdir('{}/logs'.format(folder_main))

nepoch_target = 5000
rho = 0.999
tol_step = 1e-6


planes = None
alpha = 5
clim = [0, 1]  # colour limits for plots


def save_image(x, n, f):
    misc.save_image(x.asarray(), n, f, planes=planes, clim=clim)

# load synthetic data
if not os.path.exists(folder_data):
    # simulate synthetic data
    print('folder {} does not exist. Compute it.'.format(folder_data))
    data, background, factors = mMR.simul_data()
    misc.mkdir(folder_data)
    np.save('{}/data'.format(folder_data), data)
    np.save('{}/background'.format(folder_data), background)
    np.save('{}/factors'.format(folder_data), factors)
else:
    print('folder {} exists. Load it.'.format(folder_data))
    Y = mMR.operator_mmr().range
    data = np.load('{}/data.npy'.format(folder_data), allow_pickle=True)
    data = Y.element(data)
    background = np.load('{}/background.npy'.format(folder_data), allow_pickle=True)
    background = Y.element(background)
    factors = np.load('{}/factors.npy'.format(folder_data), allow_pickle=True)
    factors = Y.element(factors)

# %%

# define operator
K = mMR.operator_mmr(factors=factors)
X = K.domain
Y = K.range
norm_K = misc.norm(K, '{}/norm_1subset.npy'.format(folder_norms))
KL = misc.kullback_leibler(Y, data, background)
folder_param = '{}/alpha{:.2g}'.format(folder_main, alpha)
misc.mkdir(folder_param)
misc.mkdir('{}/pics'.format(folder_param))

D = odl.Gradient(X)
norm_D = misc.norm(D, '{}/norm_D.npy'.format(folder_param))

c = norm_K / norm_D
D = odl.Gradient(X) * c
norm_D *= c
L1 = (alpha / c) * odl.solvers.GroupL1Norm(D.range)
L164 = (alpha / c) * odl.solvers.GroupL1Norm(D.range.astype('float64'))
g = odl.solvers.IndicatorBox(X, lower=0)

obj_fun = KL * K + L1 * D + g  # objective functional
 
# if not os.path.exists('{}/pics/gray_image_pet.png'
#                       .format(folder_param)):
#     tmp = X.element()
#     tmp_op = mMR.operator_mmr()
#     tmp_op.toodl(image, tmp)
#     fldr = '{}/pics'.format(folder_param)
#     misc.save_image(tmp.asarray(), 'image_pet', fldr, planes=planes)
#     #XXX tmp_op.toodl(image_mr, tmp)
#     # misc.save_image(tmp.asarray(), 'image_mr', fldr, planes=planes)
#     tmp_op.toodl(image_ct, tmp)
#     misc.save_image(tmp.asarray(), 'image_ct', fldr, planes=planes)
    
# %%
nepoch_target = 4000

# --- get target --- BE CAREFUL, THIS TAKES TIME
file_target = '{}/target.npy'.format(folder_param)

if True:
#if not os.path.exists(file_target):
    print('file {} does not exist. Compute it.'.format(file_target))

    A = odl.BroadcastOperator(K, D)
    f = odl.solvers.SeparableSum(KL, L1)

    norm_A = misc.norm(A, '{}/norm_tv.npy'.format(folder_main))
    sigma = rho / norm_A
    tau = rho / norm_A

    niter_target = nepoch_target

    step = 10
    cb = (CallbackPrintIteration(step=step, end=', ') &
          CallbackPrintTiming(step=step, cumulative=False, end=', ') &
          CallbackPrintTiming(step=step, cumulative=True,
                              fmt='total={:.3f} s'))

    x_opt = X.zero()
    odl.solvers.pdhg(x_opt, g, f, A, niter_target, tau, sigma,
                     callback=cb)

    obj_opt = obj_fun(x_opt)

    save_image(x_opt, 'target', '{}/pics'.format(folder_param))
    np.save(file_target, (x_opt, obj_opt))
else:
    print('file {} exists. Load it.'.format(file_target))
    x_opt, obj_opt = np.load(file_target, allow_pickle = True)
    
# %%




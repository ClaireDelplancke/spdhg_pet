#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:44:13 2020

@author: cd902

Reconstruction of map-tv problem with spdhg for different gammas
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

#%% --- load data ---

# folder structure
filename = 'map_tv'
dataset = 'amyloid'
data_suffix = 'rings0-64_span1_time3000-3600'
alpha = 5

folder_norms = '{}/norms'.format(folder_out)
folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
file_data = '{}/data_{}.npy'.format(folder_data, data_suffix)
folder_param = '{}/alpha{:.2g}'.format(folder_main, alpha)

rho = 0.999
tol_step = 1e-6

planes = None
clim = [0, 1]  # colour limits for plots


def save_image(x, n, f):
    misc.save_image(x.asarray(), n, f, planes=planes, clim=clim)


# load real data and convert to odl
(data, background, factors, image, image_mr,image_ct) = np.load(file_data, allow_pickle = True)
# (data, background, factors, image, image_ct) = np.load(file_data, allow_pickle = True)
Y = mMR.operator_mmr().range
data = Y.element(data)
background = Y.element(background)
factors = Y.element(factors)

# --- define operator --- 

K = mMR.operator_mmr(factors=factors)
X = K.domain
norm_K = misc.norm(K, '{}/norm_1subset.npy'.format(folder_norms))
KL = misc.kullback_leibler(Y, data, background)


D = odl.Gradient(X)
norm_D = misc.norm(D, '{}/norm_D.npy'.format(folder_param))

c = norm_K / norm_D
D = odl.Gradient(X) * c
norm_D *= c
L1 = (alpha / c) * odl.solvers.GroupL1Norm(D.range)
L164 = (alpha / c) * odl.solvers.GroupL1Norm(D.range.astype('float64'))
g = odl.solvers.IndicatorBox(X, lower=0)

obj_fun = KL * K + L1 * D + g  # objective functional

    
# --- get target --- 

file_target = '{}/target.npy'.format(folder_param)
x_opt, obj_opt = np.load(file_target, allow_pickle = True)
    
# %% SPDHG

# number of eppchs
nepoch = 30
# set number of subsets for algorithms
nsub = 21
# gammas
gamma_list = [0.05, 0.1, 0.5, 1.0, 5.0]

folder_today = '{}/SPDHG_new'.format(folder_param)
misc.mkdir(folder_today)
misc.mkdir('{}/npy'.format(folder_today))
misc.mkdir('{}/pics'.format(folder_today))
misc.mkdir('{}/figs'.format(folder_today))

    
# define a function to compute statistic during the iterations
class CallbackStore(odl.solvers.Callback):
    def __init__(self, name, iter_save, iter_plot, niter_per_epoch):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter_count = 0
        self.name = name
        self.out = []
        self.niter_per_epoch = niter_per_epoch

    def __call__(self, x, Kx=None, tmp=None, **kwargs):
        if type(x) is list:
            x = x[0]
        k = self.iter_count
        if k in self.iter_save:
            obj = obj_fun(x)
            psnr_opt = fom.psnr(x, x_opt)
            self.out.append({'obj': obj, 'psnr_opt': psnr_opt})
        if k in self.iter_plot:
            save_image(x, '{}_{}'.format(self.name,
                                         int(k / niter_per_epoch)),
                       '{}/pics'.format(folder_today))
        self.iter_count += 1


for gamma in gamma_list:
    file_result = '{}/npy/{}.npy'.format(folder_today, gamma)
    name = 'gamma{}'.format(gamma)

    #if False:
    if os.path.exists(file_result):
        print('file {} does exist. Do NOT compute it.'
              .format(file_result))
    else:
        print('file {} does not exist. Compute it.'
              .format(file_result))
    
        # define operators
        partition = mMR.partition_by_angle(nsub)
        tmp = mMR.operator_mmr(sino_partition=partition)
        Ys = tmp.range
        fctrs = Ys.element([factors[s, :] for s in partition])
        Ks = mMR.operator_mmr(factors=fctrs, 
                              sino_partition=partition)
    
        d = Ys.element([data[s, :] for s in partition])
        bg = Ys.element([background[s, :] for s in partition])
        KLs = misc.kullback_leibler(Ys, d, bg)
    
        norm_Ks = misc.norms(Ks, '{}/norm_{}subsets.npy'
                                 .format(folder_norms, nsub))
    
        A = odl.BroadcastOperator(*(list(Ks.operators) + [D]))
        functionals = (list(KLs.functionals) + [L1])
        f = odl.solvers.SeparableSum(*functionals)
        norm_Ai = list(norm_Ks) + [float(norm_D)]
    
        # probability
        prob = [0.5 / nsub] * nsub + [0.5]
    
        niter_per_epoch = int(np.round(nsub / sum(prob[:-1])))
        niter = nepoch * niter_per_epoch
        iter_save, iter_plot = misc.what_to_save(niter_per_epoch,
                                                 nepoch)
    
        # output function to be used with the iterations
        step = int(np.ceil(niter_per_epoch / 10))
        cb = (CallbackPrintIteration(step=step, end=', ') &
              CallbackPrintTiming(step=step, cumulative=False,
                                  end=', ') &
              CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                  cumulative=True) &
              CallbackStore(name, iter_save, iter_plot,
                            niter_per_epoch))
    
        x = X.zero()  # initialise variable
        cb(x)
    
    
    
        sigma = [rho / (nAi * gamma) for nAi in norm_Ai]
        tau_old = gamma * rho * min([pi / nAi
                         for pi, nAi in zip(prob, norm_Ai)])
        tau_new = gamma * rho / sum(norm_Ai)
        spdhg(x, f, g, A, tau_new, sigma, niter, prob=prob, callback=cb)
    
    
    
        np.save(file_result, (iter_save, niter, niter_per_epoch, x,
                              cb.callbacks[1].out, nsub, prob))

# %%  show all methods
iter_save_v, out_v, niter_per_epoch_v = {}, {}, {}
for gamma in gamma_list:
    (iter_save_v[gamma], _, niter_per_epoch_v[gamma], _, out_v[gamma], _, _
     ) = np.load('{}/npy/{}.npy'.format(folder_today, gamma), allow_pickle=True)

out = misc.resort_out(out_v, obj_opt)
misc.quick_visual_output(iter_save_v, gamma_list, out, niter_per_epoch_v,
                         folder_today)


#%% print PSNR values of interest
iter_save_v, out_v, niter_per_epoch_v = {}, {}, {}
k_epoch = 20
k_iter = 2 * nsub * k_epoch
for gamma in [0.1, 1.0]:
    (iter_save_v[gamma], _, niter_per_epoch_v[gamma], _, out_v[gamma], _, _
     ) = np.load('{}/npy/{}.npy'.format(folder_today, gamma), allow_pickle=True)
    arg = iter_save_v[gamma].index(k_iter)
    value = out_v[gamma][arg]['psnr_opt']
    print('Value of PSNR at epoch {} for gamma = {} is {}'.format(k_epoch, gamma, value))

# %% Compute OSEM reconstruction
#    Breaks OSEM

# name = 'OSEM'
# folder_today = '{}/OSEM'.format(folder_param)
# misc.mkdir(folder_today)
# misc.mkdir('{}/npy'.format(folder_today))
# misc.mkdir('{}/pics'.format(folder_today))
# misc.mkdir('{}/figs'.format(folder_today))
# file_result = '{}/npy/{}.npy'.format(folder_today, name)
# nsub = 21
# step = nsub
# niter_per_epoch = nsub
# nepoch = 20
# niter = nepoch * niter_per_epoch
# partition = mMR.partition_by_bin(nsub)
# Ys = mMR.operator_mmr(sino_partition=partition).range
# fctrs = Ys.element([factors[s, :] for s in partition])
# d = Ys.element([data[s, :] for s in partition])
# bg = Ys.element([background[s, :] for s in partition])

# # define operator
# Ks = mMR.operator_mmr(factors=fctrs, sino_partition=partition)
# # data fit
# KLs = misc.kullback_leibler(Ys, d, bg)  

# iter_save, iter_plot = misc.what_to_save(niter_per_epoch,
#                                                  nepoch)
# #callback
# cb = (CallbackPrintIteration(step=step, end=', ') &
#       CallbackPrintTiming(step=step, cumulative=False, end=', ') &
#       CallbackPrintTiming(step=step, fmt='total={:.3f} s',
#                           cumulative=True) &
#       CallbackStore(name, iter_save, iter_plot,
#                     niter_per_epoch))
    
# x = X.one()  # initialise variable
# x= misc.OSEM(x, KLs.data, KLs.background, Ks, niter, callback=cb)

# prob = [1]

# np.save(file_result, (iter_save, niter, niter_per_epoch, x,
#                               cb.callbacks[1].out, nsub, prob))


# %% SYNTHETIC DATA

folder_data = '/home/cd902/siemens-biograph_data/synthetic'
folder_out = '/u/s/cd902/GitHub/spdhg_pet/results/synthetic'

#%% --- load data ---

# folder structure
filename = 'map_tv'
dataset = 'synthetic'

alpha = 5

folder_norms = '{}/norms'.format(folder_out)
folder_main = '{}/{}_{}'.format(folder_out, filename, dataset)
folder_param = '{}/alpha{:.2g}'.format(folder_main, alpha)

rho = 0.999
tol_step = 1e-6

planes = None
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

# --- define operator --- 

K = mMR.operator_mmr(factors=factors)
X = K.domain
Y = K.range
norm_K = misc.norm(K, '{}/norm_1subset.npy'.format(folder_norms))
KL = misc.kullback_leibler(Y, data, background)


D = odl.Gradient(X)
norm_D = misc.norm(D, '{}/norm_D.npy'.format(folder_param))

c = norm_K / norm_D
D = odl.Gradient(X) * c
norm_D *= c
L1 = (alpha / c) * odl.solvers.GroupL1Norm(D.range)
L164 = (alpha / c) * odl.solvers.GroupL1Norm(D.range.astype('float64'))
g = odl.solvers.IndicatorBox(X, lower=0)

obj_fun = KL * K + L1 * D + g  # objective functional

    
# --- get target --- 

file_target = '{}/target.npy'.format(folder_param)
x_opt, obj_opt = np.load(file_target, allow_pickle = True)
    
# %% SPDHG

# number of eppchs
nepoch = 30
# set number of subsets for algorithms
nsub = 21
# gammas
gamma_list = [0.05, 0.1, 0.5, 1.0, 5.0]

folder_today = '{}/SPDHG_new2'.format(folder_param)
misc.mkdir(folder_today)
misc.mkdir('{}/npy'.format(folder_today))
misc.mkdir('{}/pics'.format(folder_today))
misc.mkdir('{}/figs'.format(folder_today))

    
# define a function to compute statistic during the iterations
class CallbackStore(odl.solvers.Callback):
    def __init__(self, name, iter_save, iter_plot, niter_per_epoch):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter_count = 0
        self.name = name
        self.out = []
        self.niter_per_epoch = niter_per_epoch

    def __call__(self, x, Kx=None, tmp=None, **kwargs):
        if type(x) is list:
            x = x[0]
        k = self.iter_count
        if k in self.iter_save:
            obj = obj_fun(x)
            psnr_opt = fom.psnr(x, x_opt)
            self.out.append({'obj': obj, 'psnr_opt': psnr_opt})
        if k in self.iter_plot:
            save_image(x, '{}_{}'.format(self.name,
                                         int(k / niter_per_epoch)),
                       '{}/pics'.format(folder_today))
        self.iter_count += 1


for gamma in gamma_list:
    file_result = '{}/npy/{}.npy'.format(folder_today, gamma)
    name = 'gamma{}'.format(gamma)

    #if False:
    if os.path.exists(file_result):
        print('file {} does exist. Do NOT compute it.'
              .format(file_result))
    else:
        print('file {} does not exist. Compute it.'
              .format(file_result))
    
        # define operators
        partition = mMR.partition_by_angle(nsub)
        tmp = mMR.operator_mmr(sino_partition=partition)
        Ys = tmp.range
        fctrs = Ys.element([factors[s, :] for s in partition])
        Ks = mMR.operator_mmr(factors=fctrs, 
                              sino_partition=partition)
    
        d = Ys.element([data[s, :] for s in partition])
        bg = Ys.element([background[s, :] for s in partition])
        KLs = misc.kullback_leibler(Ys, d, bg)
    
        norm_Ks = misc.norms(Ks, '{}/norm_{}subsets.npy'
                                 .format(folder_norms, nsub))
    
        A = odl.BroadcastOperator(*(list(Ks.operators) + [D]))
        functionals = (list(KLs.functionals) + [L1])
        f = odl.solvers.SeparableSum(*functionals)
        norm_Ai = list(norm_Ks) + [float(norm_D)]
    
        # probability
        prob = [0.5 / nsub] * nsub + [0.5]
    
        niter_per_epoch = int(np.round(nsub / sum(prob[:-1])))
        niter = nepoch * niter_per_epoch
        iter_save, iter_plot = misc.what_to_save(niter_per_epoch,
                                                 nepoch)
    
        # output function to be used with the iterations
        step = int(np.ceil(niter_per_epoch / 10))
        cb = (CallbackPrintIteration(step=step, end=', ') &
              CallbackPrintTiming(step=step, cumulative=False,
                                  end=', ') &
              CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                  cumulative=True) &
              CallbackStore(name, iter_save, iter_plot,
                            niter_per_epoch))
    
        x = X.zero()  # initialise variable
        cb(x)
    
    
    
        sigma = [rho / (nAi * gamma) for nAi in norm_Ai]
        tau_old = gamma * rho * min([pi / nAi
                         for pi, nAi in zip(prob, norm_Ai)])
        tau_new = gamma * rho / sum(norm_Ai)
        spdhg(x, f, g, A, tau_new, sigma, niter, prob=prob, callback=cb)
    
    
    
        np.save(file_result, (iter_save, niter, niter_per_epoch, x,
                              cb.callbacks[1].out, nsub, prob))

# %%  show all methods
iter_save_v, out_v, niter_per_epoch_v = {}, {}, {}
name_list = ['{}'.format(gamma) for gamma in gamma_list]
for gamma in gamma_list:
    (iter_save_v['{}'.format(gamma)], _, niter_per_epoch_v['{}'.format(gamma)], _, out_v['{}'.format(gamma)], _, _
     ) = np.load('{}/npy/{}.npy'.format(folder_today, gamma), allow_pickle=True)

out = misc.resort_out(out_v, obj_opt)
misc.quick_visual_output(iter_save_v, name_list, out, niter_per_epoch_v,
                         folder_today)


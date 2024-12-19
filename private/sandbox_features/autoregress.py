#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:07:16 2024

@author: balestrieri
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import re
import os, fnmatch, sys

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test


from scipy.fft import fft, fftfreq

#%% 

def make_same_length(A, B):
    
    l_vects = np.array([len(A), len(B)])
    min_l = l_vects.min()
    
    return A[0:min_l], B[0:min_l]


def apply_FFT(vec, fsample):
    
    N = len(vec)
    T = 1/fsample
    yf = np.abs(fft(vec, axis=0)[0:N//2])
    xf = fftfreq(N, T)[:N//2]
    
    return xf, yf

#%%
# srate = 250 Hz

EC = loadmat('EC_dat')['EC_dat']
EO = loadmat('EO_dat')['EO_dat']

EC, EO = make_same_length(EC, EO)

N = len(EC)

mdl_EC = AutoReg(EC, 250)
res_EC = mdl_EC.fit()

mdl_EO = AutoReg(EO, 250)
res_EO = mdl_EO.fit()

print(res_EO.summary())

out_EC = res_EC.predict(start=N, end=N + 9999)
out_EO = res_EO.predict(start=N, end=N + 9999)

out_EC = out_EC[~np.isnan(out_EC)]
out_EO = out_EO[~np.isnan(out_EO)]

#%% compute spectra and compare spectra from autoregress and original

ntrls = 100

xf_dat, fftEC_dat = apply_FFT(np.reshape(EC[:, 0], (len(EC)//ntrls, ntrls), order='F'), fsample=250)
xf_dat, fftEO_dat = apply_FFT(np.reshape(EO, (len(EO)//ntrls, ntrls), order='F'), fsample=250)

xf_sim, fftEC_sim = apply_FFT(np.reshape(out_EC, (len(out_EC)//ntrls, ntrls), order='F'), fsample=250)
xf_sim, fftEO_sim = apply_FFT(np.reshape(out_EO, (len(out_EO)//ntrls, ntrls), order='F'), fsample=250)


#%%

plt.figure()
plt.subplot(121)
plt.plot(xf_dat, fftEC_dat.mean(axis=1))
plt.plot(xf_dat, fftEO_dat.mean(axis=1))

plt.subplot(122)
plt.plot(xf_sim, fftEC_sim.mean(axis=1))
plt.plot(xf_sim, fftEO_sim.mean(axis=1))


#%% evaluate correlation between power and std/mad

#input folder 
infold = '../STRG_computed_features/LEMON/'

mdltypes = ['gammaCompare']

# concatenate files between participant, after within-participant normalization
acc_type = 0
full_count_exc = 0

# compile regex for further PC selection in keys
r = re.compile("PC_full_set") 
        
for imdl in mdltypes:
    
    data_type = imdl
    filt_fnames = fnmatch.filter(os.listdir(infold), 'sub-*' + imdl + '*.mat')
    acc = 0
    for iname in filt_fnames: 
        iname = iname[0:10]
        filt_fnames[acc] = iname
        acc +=1
    
    fullX_train, fullX_test, Y_train, Y_test, subjID_trials_labels = cat_subjs_train_test(infold, subjlist=filt_fnames, 
                                                            ftype=data_type, tanh_flag=True, 
                                                            compress_flag=True,
                                                            pca_kept_var=.9)
    
#%% 1st comp for mad and power

alpha_mad = fullX_train['PC_alpha_8_13_Hz_mad'][:, 0]
alpha_power = fullX_train['PC_alpha_power_8_13_Hz'][:, 0]
npol = 2
x = np.linspace(-5, 5, 1000)

EC_pwr = alpha_power[Y_train==1]
EO_pwr = alpha_power[Y_train==2]
EC_mad = alpha_mad[Y_train==1]
EO_mad = alpha_mad[Y_train==2]

plt.figure()
plt.subplot(221)
plt.scatter(x=EC_mad, y=EC_pwr, s=.001)
plt.scatter(x=EO_mad, y=EO_pwr, s=.001)
plt.xlabel('alpha power')
plt.ylabel('alpha mad')


plt.subplot(222)

z_EC = np.polyfit(EC_mad, EC_pwr, npol)
poly_mdl_EC = np.poly1d(z_EC)
plt.scatter(x=EC_mad, y=EC_pwr, s=.001)
plt.plot(x, poly_mdl_EC(x), c='r')
plt.xlabel('alpha power')
plt.ylabel('alpha mad')

plt.subplot(223)

z_EO = np.polyfit(EO_mad, EO_pwr, npol)
poly_mdl_EO = np.poly1d(z_EO)
plt.plot(x, poly_mdl_EO(x), c='r')
plt.scatter(x=EO_mad, y=EO_pwr, s=.001)
plt.xlabel('alpha power')
plt.ylabel('alpha mad')

plt.subplot(224)

plt.plot(x, poly_mdl_EC(x))
plt.plot(x, poly_mdl_EO(x))
plt.legend(('EC', 'EO'))

R_EC = np.corrcoef(EC_mad, EC_pwr)
print(R_EC)
R_EO = np.corrcoef(EO_mad, EO_pwr)
print(R_EO)

#%% median filter

from scipy.signal import medfilt

arr1inds = EC_pwr.argsort()
EC_srtd_mad = EC_mad[arr1inds]
EC_srtd_pwr = EC_pwr[arr1inds]

EC_filt_pwr = medfilt(EC_srtd_pwr, kernel_size=301)
EC_filt_mad = medfilt(EC_srtd_mad, kernel_size=301)


arr2inds = EO_pwr.argsort()
EO_srtd_mad = EO_mad[arr2inds]
EO_srtd_pwr = EO_pwr[arr2inds]

EO_filt_pwr = medfilt(EO_srtd_pwr, kernel_size=301)
EO_filt_mad = medfilt(EO_srtd_mad, kernel_size=301)


plt.figure()
plt.plot(EC_filt_pwr, EC_filt_mad)
plt.plot(EO_filt_pwr, EO_filt_mad)


#%% 


alpha_mad = fullX_test['PC_alpha_8_13_Hz_mad'][:, 0]
alpha_power = fullX_test['PC_alpha_8_13_Hz_std'][:, 0]

# alpha_power = fullX_test['PC_alpha_power_8_13_Hz'][:, 0]

plt.figure()
plt.subplot(131)
plt.scatter(x=alpha_mad[Y_test==1], y=alpha_power[Y_test==1], s=.001)
plt.scatter(x=alpha_mad[Y_test==2], y=alpha_power[Y_test==2], s=.001)

plt.subplot(132)
plt.scatter(x=alpha_mad[Y_test==1], y=alpha_power[Y_test==1], s=.001)

plt.subplot(133)
plt.scatter(x=alpha_mad[Y_test==2], y=alpha_power[Y_test==2], s=.001)

R_EC = np.corrcoef(alpha_mad[Y_test==1], alpha_power[Y_test==1])
print(R_EC)
R_EO = np.corrcoef(alpha_mad[Y_test==2], alpha_power[Y_test==2])
print(R_EO)








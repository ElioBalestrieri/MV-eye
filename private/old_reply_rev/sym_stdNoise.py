#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:50:14 2024

@author: balestrieri
"""


import pyplnoise
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# scaler
from sklearn.preprocessing import RobustScaler

# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#%%

fs = 500; N = 500
ntrls = 10000
PinkNoiseGen = pyplnoise.AlphaNoise(fs, .01, fs/2., alpha=1.5, seed=42)
WhiteNoiseGen = pyplnoise.WhiteNoise(fs, seed=42)


P_times, W_times = [], []
for itrl in range(ntrls):
    P_times.append(PinkNoiseGen.get_series(N))
    W_times.append(WhiteNoiseGen.get_series(N))
    
pinkNoise_TS = np.array(P_times).T
whiteNoise_TS = np.array(W_times).T #  np.random.randn(N, ntrls)

#%%

# standard deviation
pinkSTD = np.std(pinkNoise_TS, axis=0)
whiteSTD = np.std(whiteNoise_TS, axis=0)

pinkVAR = np.var(pinkNoise_TS, axis=0)
whiteVAR = np.var(whiteNoise_TS, axis=0)


# FFT
pinkFFT = np.abs(sp.fft.fft(pinkNoise_TS, axis=0)[1:N//2, :]).sum(axis=0) # remove DC comp?
whiteFFT = np.abs(sp.fft.fft(whiteNoise_TS, axis=0)[1:N//2, :]).sum(axis=0)

# pinkFFT = (np.abs(sp.fft.fft(pinkNoise_TS, axis=0)[1:N//2, :])**2).sum(axis=0) # remove DC comp?
# whiteFFT = (np.abs(sp.fft.fft(whiteNoise_TS, axis=0)[1:N//2, :])**2).sum(axis=0)


# PSD -welch
f, pinkPSD = sp.signal.welch(pinkNoise_TS.T, fs=fs)
f, whitePSD = sp.signal.welch(whiteNoise_TS.T, fs=fs)

# scale
sclr_mdl = RobustScaler()
pinkSTD_scaled = sclr_mdl.fit_transform(pinkSTD[:, np.newaxis])[:, 0]
sclr_mdl = RobustScaler()
whiteSTD_scaled = sclr_mdl.fit_transform(whiteSTD[:, np.newaxis])[:, 0]
sclr_mdl = RobustScaler()
pinkFFT_scaled = sclr_mdl.fit_transform(pinkFFT[:, np.newaxis])[:, 0]
sclr_mdl = RobustScaler()
whiteFFT_scaled = sclr_mdl.fit_transform(whiteFFT[:, np.newaxis])[:, 0]
sclr_mdl = RobustScaler()
pinkVAR_scaled = sclr_mdl.fit_transform(pinkVAR[:, np.newaxis])[:, 0]
sclr_mdl = RobustScaler()
whiteVAR_scaled = sclr_mdl.fit_transform(whiteVAR[:, np.newaxis])[:, 0]

# fit chi dist to FFT
mdl = sp.stats.chi2.fit(pinkFFT)
ECDF = sp.stats.chi2.cdf(pinkFFT, df=mdl[0], loc=mdl[1], scale=mdl[2])
Gauss_vals = sp.stats.norm.ppf(ECDF)


#%%

# plt.figure()
#plt.subplot(121)
sns.jointplot(x=pinkFFT_scaled, y=pinkSTD_scaled, size=.1, alpha=.2)
plt.xlabel('sum FFT')
plt.ylabel('std')

mdl_reg = LinearRegression()
mdl_reg.fit(pinkFFT_scaled[:, np.newaxis], pinkSTD_scaled[:, np.newaxis])
R2_pink = r2_score(pinkSTD_scaled[:, np.newaxis], mdl_reg.predict(pinkFFT_scaled[:, np.newaxis]))
plt.suptitle('Pink Noise\nR2 : {:0.4}'.format(R2_pink))
plt.legend([],[], frameon=False)
plt.tight_layout()

# plt.figure()
# plt.subplot(122)
sns.jointplot(x=whiteFFT_scaled, y=whiteSTD_scaled, size=.1, alpha=.2)
plt.xlabel('sum FFT')
plt.ylabel('std')

mdl_reg = LinearRegression()
mdl_reg.fit(whiteFFT_scaled[:, np.newaxis], whiteSTD_scaled[:, np.newaxis])
R2_white = r2_score(whiteSTD_scaled[:, np.newaxis], mdl_reg.predict(whiteFFT_scaled[:, np.newaxis]))
plt.suptitle('White Noise\nR2 : {:0.4}'.format(R2_white))
plt.legend([],[], frameon=False)
plt.tight_layout()


#%%

# plt.figure()
#plt.subplot(121)
sns.jointplot(x=pinkFFT_scaled, y=pinkVAR_scaled, size=.1, alpha=.2)
plt.xlabel('sum FFT')
plt.ylabel('var')

mdl_reg = LinearRegression()
mdl_reg.fit(pinkFFT_scaled[:, np.newaxis], pinkVAR_scaled[:, np.newaxis])
R2_pink = r2_score(pinkVAR_scaled[:, np.newaxis], mdl_reg.predict(pinkFFT_scaled[:, np.newaxis]))
plt.suptitle('Pink Noise\nR2 : {:0.4}'.format(R2_pink))
plt.legend([],[], frameon=False)
plt.tight_layout()

# plt.figure()
# plt.subplot(122)
sns.jointplot(x=whiteFFT_scaled, y=whiteVAR_scaled, size=.1, alpha=.2)
plt.xlabel('sum FFT')
plt.ylabel('var')

mdl_reg = LinearRegression()
mdl_reg.fit(whiteFFT_scaled[:, np.newaxis], whiteVAR_scaled[:, np.newaxis])
R2_white = r2_score(whiteVAR_scaled[:, np.newaxis], mdl_reg.predict(whiteFFT_scaled[:, np.newaxis]))
plt.suptitle('White Noise\nR2 : {:0.4}'.format(R2_white))
plt.legend([],[], frameon=False)
plt.tight_layout()

#%%

new_N = 499

pinkNoise_TS = np.array(P_times).T
pinknoise_der = np.diff(pinkNoise_TS, axis=0)


ts_examples = {'PN_orig' : sp.stats.zscore(pinkNoise_TS[:, 0])[1:],
               'PN_der' : sp.stats.zscore(pinknoise_der[:, 0])}

ts_examples = {'PN_orig' : pinkNoise_TS[1:, 0],
               'PN_der' : pinknoise_der[:, 0]}

FFT_examples = {'PN_orig' : np.abs(sp.fft.fft(pinkNoise_TS[1:, 0])[1:new_N//2])**2,
               'PN_der' : np.abs(sp.fft.fft(pinknoise_der[:, 0])[1:new_N//2])**2}

freqs = np.fft.fftfreq(new_N, d=1/250)[1:new_N//2]
times = np.arange(1/250, 2, step=1/250)

plt.figure()
plt.subplot(211)
plt.plot(times, ts_examples['PN_orig'])
plt.plot(times, ts_examples['PN_der'])
plt.xlabel('time (s)')
plt.title('Time domain')
plt.legend(['original', 'derivative'])

plt.subplot(212)
plt.plot(np.log(freqs), np.log(FFT_examples['PN_orig']))
plt.plot(np.log(freqs), np.log(FFT_examples['PN_der']))
plt.title('Frequency domain')
plt.ylabel('log power')
plt.xlabel('log freq')

plt.tight_layout()

#%%

pinkFFT = np.abs(sp.fft.fft(pinkNoise_TS, axis=0)[1:N//2, :]).sum(axis=0) # remove DC comp?

# plt.figure()

# plt.subplot(121)
# sns.scatterplot(x=pinkPSD_scaled, y=pinkSTD_scaled, size=.1, alpha=.2)
# plt.xlabel('sum PSD')
# plt.ylabel('std')

# mdl_reg = LinearRegression()
# mdl_reg.fit(pinkPSD_scaled[:, np.newaxis], pinkSTD_scaled[:, np.newaxis])
# R2_pink = r2_score(pinkSTD_scaled[:, np.newaxis], mdl_reg.predict(pinkPSD_scaled[:, np.newaxis]))
# plt.title('Pink Noise\nR2 : {:0.4}'.format(R2_pink))
# plt.legend([],[], frameon=False)


# plt.subplot(122)
# sns.scatterplot(x=whitePSD_scaled, y=whiteSTD_scaled, size=.1, alpha=.2)
# plt.xlabel('sum PSD')
# plt.ylabel('std')

# mdl_reg = LinearRegression()
# mdl_reg.fit(whitePSD_scaled[:, np.newaxis], whiteSTD_scaled[:, np.newaxis])
# R2_white = r2_score(whiteSTD_scaled[:, np.newaxis], mdl_reg.predict(whitePSD_scaled[:, np.newaxis]))
# plt.title('White Noise\nR2 : {:0.4}'.format(R2_white))
# plt.legend([],[], frameon=False)

#%%

x_time = np.arange(0, 1, 1/256)
y_sin = np.sin(x_time*2*np.pi*10)






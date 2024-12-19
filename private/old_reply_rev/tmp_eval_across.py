#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:09:40 2024

@author: balestrieri
"""




infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/rev_reply_MEG/'

import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns


#%%

# plt.legend([],[], frameon=False)




# plt.figure()
# sns.violinplot(data=DF_WSscaled, y='accuracy', x='model',
#                   palette="ch:start=.2,rot=-.3, dark=.4", fill=False,
#                   split=True, hue='condition', cut=0)

# plt.title('across subject accuracy\nSVM, scaled WS', fontsize=16)

#%%

DF_ECEO = DF_WSscaled.loc[DF_WSscaled['condition']=='ECEO', :]
DF_VS = DF_WSscaled.loc[DF_WSscaled['condition']=='VS', :]

plt.figure()
plt.subplot(121)
sns.violinplot(data=DF_ECEO, y='accuracy', x='model', fill=False,
               split=True, hue='classifier', cut=0)

plt.title('ECEO', fontsize=16)
plt.legend([],[], frameon=False)


plt.subplot(122)
sns.violinplot(data=DF_VS, y='accuracy', x='model', fill=False,
               split=True, hue='classifier', cut=0)

plt.title('VS', fontsize=16)





#%%



# ANOVA condiitons
aov_Mdl_across = pg.rm_anova(data=DFfull, dv='accuracy', 
                             within=['model', 'condition'], subject='cvFolds',
                             detailed=True)        
print('\nANOVA across (full)')
print(aov_Mdl_across.round(3).to_string())

#%% subplot full set across
plt.figure()
ax = sns.violinplot(data=DFfull, y='accuracy', x='model',
                  palette="ch:start=.2,rot=-.3, dark=.4", fill=False,
                  split=True, hue='condition',  cut=0)

plt.title('across subject accuracy', fontsize=16)
# plt.legend([],[], frameon=False)
plt.ylabel('balanced accuracy', fontsize=12)
plt.xlabel('Feature set', fontsize=12)


#%%

aov_Mdl_across = pg.rm_anova(data=DFfull, dv='accuracy', 
                             within=['classifier', 'condition'], subject='cvFolds',
                             detailed=True)        
print('\nANOVA across (full)')
print(aov_Mdl_across.round(3).to_string())

plt.figure()
ax = sns.violinplot(data=DFfull, y='accuracy', x='classifier',
                  palette="ch:start=.2,rot=-.3, dark=.4_r", fill=False,
                  split=True, hue='condition', cut=0)

plt.title('across subject accuracy', fontsize=16)
# plt.legend([],[], frameon=False)
# plt.ylabel('balanced accuracy', fontsize=12)
# plt.xlabel('Feature set', fontsize=12)


#%% SVM only


#%% interaction between model & classifiers

plt.figure()
ax = sns.violinplot(data=DFfull, y='accuracy', x='model',
                   fill=False,
                   split=True, hue='classifier', cut=0)

plt.title('across subject accuracy\nmodel vs classifier', fontsize=16)



#%% drift-based measures

drift_sets_mask = ((DF['feature'] == 'median') | (DF['feature'] == 'mean')) & (DF['classifier'] == 'SVM')
DF_drift = DF.loc[drift_sets_mask, :]

plt.figure()
ax = sns.violinplot(data=DF_drift, y='accuracy', x='feature',
                    palette="ch:start=.2,rot=-.3, dark=.4_r",
                    fill=False,
                    split=True, hue='condition', cut=0)

plt.title('drift measures', fontsize=16)



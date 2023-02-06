#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:42:52 2023

@author: elio
"""

#%% import and folder definition

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

infold = '../STRG_decoding_accuracy/'
plt.close('all')

#%% fetch files

up_to_subj = 22

for isubj in range(up_to_subj):
    
    subjname = f'{isubj+1:02d}'
    fname_bands = infold + subjname + '_freqBands.csv'
    fname_full = infold + subjname + '_nobp.csv'
    DF_freqbands = pd.read_csv(fname_bands)
    DF_full = pd.read_csv(fname_full)
    DF_full.loc[0, 'Unnamed: 0'] = 'all_bands'

    # long format needed because difference in columns between full freq and bandpassed signal
    long_DF_bands = DF_freqbands.set_index('Unnamed: 0').stack().to_frame().\
                    reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                  'level_1': 'feature', 
                                                  0: 'accuracy'})
    long_DF_full =  DF_full.set_index('Unnamed: 0').stack().to_frame().\
                    reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                  'level_1': 'feature', 
                                                  0: 'accuracy'})

    DF_subj = pd.concat([long_DF_bands, long_DF_full], ignore_index=True)        
    DF_subj.insert(0, 'SubjID', subjname)
    
    if isubj == 0:
        allsubjs_DF = DF_subj.copy()
    else:
        allsubjs_DF = pd.concat([allsubjs_DF, DF_subj])
        
        
#%% plotting
rounded_DF = allsubjs_DF.round(3)

# 1. plot the full classifiers accuracies
# slice only full_set feature in all frequency bands -including no bandpass
plt.figure()
fullset_DF = rounded_DF.loc[rounded_DF['feature']=='full_set', :]
ax = sns.barplot(data=fullset_DF, y='freq_band', x='accuracy', orient='h',
                 palette='Set3')
plt.tight_layout()
plt.title('SVM, full features set, frequency bands')
plt.xlabel('balanced accuracy')
plt.xlim((.5, 1))

# 2. for each freq band plot the ordered full set of features.
# moreover, select the best 3 features in each freq band
fbands_names = rounded_DF.freq_band.unique()

acc_band = 0
for iband in fbands_names:

    plt.figure()     
    this_DF = rounded_DF.loc[rounded_DF['freq_band']==iband, :]

    # go to wide format to allow sorting
    wide_DF = pd.pivot(this_DF, index='SubjID', columns='feature', 
                       values='accuracy')    
    new_idx = wide_DF.mean().sort_values(ascending=False)
    wide_DF = wide_DF.reindex(new_idx.index, axis=1)
    
    # plot
    ax = sns.barplot(data=wide_DF, orient='h',
                     palette="ch:start=.2,rot=-.3, dark=.4")
    
    plt.tight_layout()
    plt.title('SVM acc, ' + iband)
    plt.xlabel('balanced accuracy')
    plt.xlim((.5, 1))
    
    # collect either the top 2 features OR all the features > .90 
    # for the current bandpass condition
    winning_feats = new_idx.index[new_idx>.9]
    if winning_feats.empty:
        winning_feats = new_idx.index[0:2]
    
    this_band_best_feats = this_DF[this_DF.feature.isin(winning_feats)]

    cols_bind = ['freq_band', 'feature']; swap_feats = this_band_best_feats.copy()
    swap_feats['combined'] = this_band_best_feats[cols_bind].apply(
        lambda row: '\n'.join(row.values.astype(str)), axis=1)
    
    if acc_band == 0:
        allsubjs_and_bands_best = swap_feats.copy()
    else:
        allsubjs_and_bands_best = pd.concat([allsubjs_and_bands_best, 
                                             swap_feats])

    acc_band += 1

#%% 3. plot the best feats across subjects and bands

# usual long to wide
best_feats_wide = pd.pivot(allsubjs_and_bands_best, index='SubjID', 
                           columns='combined', values='accuracy')    
new_idx = best_feats_wide.mean().sort_values(ascending=False)
best_feats_sorted = best_feats_wide.reindex(new_idx.index, axis=1)

plt.figure()

ax = sns.stripplot(data=best_feats_sorted, orient='h', 
                   palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
ax = sns.barplot(data=best_feats_sorted, orient='h', errcolor=(.3, .3, .3, 1),
    linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))

plt.tight_layout()
plt.title('Winning features, (within subjects)')
plt.xlabel('balanced accuracy')
plt.ylabel('freq band / feature')
plt.xlim((.6, 1))

cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=len(new_idx))
acc_ypos = .15; acc_color = 0
for itxt in new_idx:
    
    plt.text(.6, acc_ypos, str(round(itxt, 3)), color=cpal[acc_color])
    acc_ypos += 1; acc_color += 1




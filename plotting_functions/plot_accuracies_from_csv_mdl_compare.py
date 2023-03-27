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
import numpy as np
import pingouin as pg

infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/'
plt.close('all')

#%% single subjects

ExpConds = ['ECEO', 'VS'] 
MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats']

up_to_subj = 29

allsubjs_within_DFs, allsubjs_between_DFs = [], []

for isubj in range(up_to_subj):

    for iExp in ExpConds:
        
        for iMdl in MdlTypes:
    
            subjname = f'{isubj+1:02d}'
    
            # VS missing within
            if iExp != 'VS':
                
                # within subjects            
                fname_within = infold + subjname + '_' + iExp + '_' + iMdl + '_feats_reduced.csv'
                DF_WN = pd.read_csv(fname_within)
                
                tmp = DF_WN.set_index('Unnamed: 0').stack().to_frame().\
                            reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                          'level_1': 'feature', 
                                                          0: 'decoding_accuracy'})
         
                tmp = tmp.drop('freq_band', axis=1)
                
                # add columns
                tmp['subjID'] = [subjname] * tmp.shape[0]
                tmp['ExpCond'] = [iExp] * tmp.shape[0]
                tmp['MdlType'] = [iMdl] * tmp.shape[0]
    
                allsubjs_within_DFs.append(tmp)

            # between subjects
            fname_between = infold + 'ID_' + subjname + '_leftout_' + iExp + '_' + iMdl + '_intersubjs_accs.csv'
            DF_BW = pd.read_csv(fname_between)

            tmp = DF_BW.set_index('Unnamed: 0').stack().to_frame().\
                        reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                      'level_1': 'feature', 
                                                      0: 'decoding_accuracy'})
     
            tmp = tmp.drop('freq_band', axis=1)
            
            # add columns
            tmp['subjID'] = [subjname] * tmp.shape[0]
            tmp['ExpCond'] = [iExp] * tmp.shape[0]
            tmp['MdlType'] = [iMdl] * tmp.shape[0]

            allsubjs_between_DFs.append(tmp)
                
            
            
DF_fullsample_WN = pd.concat(allsubjs_within_DFs, ignore_index=True) 
DF_fullsample_BW = pd.concat(allsubjs_between_DFs, ignore_index=True) 
    
    
#%% stats

# 1. winning "model": fullFFT, vs time feats, vs freq bands
DF_full_set_WN = DF_fullsample_WN.loc[DF_fullsample_WN['feature']=='full_set', :]

aov_winningMdl = pg.rm_anova(data=DF_full_set_WN, dv='decoding_accuracy', 
                             within='MdlType', subject='subjID', detailed=True)    
    
aov_winningMdl.round(3)    
            
ttest_winningMdl = pg.ttest(DF_full_set_WN['decoding_accuracy'].loc[DF_full_set_WN['MdlType']=='FreqBands'], 
                            DF_full_set_WN['decoding_accuracy'].loc[DF_full_set_WN['MdlType']=='TimeFeats'], 
                            paired=True)
print(ttest_winningMdl)

plt.figure()
ax = sns.barplot(data=DF_full_set_WN.round(3), y='decoding_accuracy', x='MdlType',
            palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")
# sns.move_legend(ax, "lower center",
#     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)
plt.ylim((.7, 1))
# plt.tight_layout()

plt.suptitle('Features type, within subject accuracy')


# 2. between 
DF_full_set_BW = DF_fullsample_BW.loc[DF_fullsample_BW['feature']=='aggregate', :]


plt.figure()
ax = sns.barplot(data=DF_fullsample_BW, y='decoding_accuracy', x='MdlType',
            hue='ExpCond', palette="ch:start=.2,rot=-.3, dark=.4")
plt.ylim((.5, 1))

plt.suptitle('Features type, between subjects (LOO) accuracy')




#%% fetch 

freq_bands_names = ['delta_1_4_Hz', 'theta_4_8_Hz', 'alpha_8_13_Hz',
                    'beta_13_30_Hz', 'low_gamma_30_45_Hz', 'high_gamma_55_100_Hz',
                    'feats']

for iband in freq_bands_names:
    
    fname = infold + iband + '_intersubjs_accs.csv'
    DF_freqband_across_subjs = pd.read_csv(fname)
    long_DF_AS = DF_freqband_across_subjs.set_index('Unnamed: 0').transpose()
    
    if iband == 'feats':
        this_band = 'all_bands'
    else:
        this_band = iband
        
    idxs_feats = long_DF_AS.index
    
    for this_feat in idxs_feats:
        
        if this_feat == 'aggregate':
            old_feat_name = 'full_set'
        else:
            old_feat_name = this_feat
        
        tmp_ = long_DF_AS.loc[this_feat, :]
        bnd_mask = comparative_accs_DF['freq_band']==this_band
        feat_mask = comparative_accs_DF['feature']==old_feat_name

        comparative_accs_DF.loc[bnd_mask&feat_mask,'CV_BS_acc'] = tmp_['CV_accuracy'] 
        comparative_accs_DF.loc[bnd_mask&feat_mask,'LO_acc'] = tmp_['GEN_accuracy'] 
        
    
#%% compute average accuracy and sort accordingly

comparative_accs_DF['AVG_acc'] = comparative_accs_DF.iloc[:, 2:5].mean(axis=1)
comparative_accs_DF = comparative_accs_DF.sort_values('AVG_acc', ascending=False)
    
# plot the 10 best features across all
winning_feats_allconds = comparative_accs_DF.iloc[0:10, :]
cols_bind = ['freq_band', 'feature']; 
winning_feats_allconds['features'] = winning_feats_allconds[cols_bind].apply(
        lambda row: '\n'.join(row.values.astype(str)), axis=1)

winning_feats_allconds = winning_feats_allconds.drop(['freq_band', 'feature', 
                                                      'AVG_acc'], axis=1)
    
out = pd.melt(winning_feats_allconds, id_vars='features', value_name='decoding accuracy', 
              value_vars=['CV_WS_acc', 'CV_BS_acc', 'LO_acc'])

plt.figure()
ax = sns.barplot(data=out.round(3), x='decoding accuracy', y='features',
            hue='variable', orient='h', palette="ch:start=.2,rot=-.3, dark=.4")
sns.move_legend(ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)
plt.xlim((.7, 1))
plt.tight_layout()

plt.suptitle('Winning features')

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

#%% 3. plot the best feats within subjects and bands

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

ax.set_yticklabels(best_feats_sorted.columns, size=8)


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



plt.show()
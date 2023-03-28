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
    
    
#%% stats & plots
# 1. compare the full set of features in 3 different conditions: fullFFT, FreqBands
# TimeFeats

# 1. winning "model": fullFFT, vs time feats, vs freq bands

# 1a: within subjects accuracy
DF_full_set_WN = DF_fullsample_WN.loc[DF_fullsample_WN['feature']=='full_set', :]
aov_Mdl_WN = pg.rm_anova(data=DF_full_set_WN, dv='decoding_accuracy', 
                              within=['MdlType', 'ExpCond'], subject='subjID',
                              detailed=False)        
aov_Mdl_WN.round(3)
            
ttest_FreqTime_WN = pg.ttest(DF_full_set_WN['decoding_accuracy'].loc[
                                        DF_full_set_WN['MdlType']=='FreqBands'], 
                            DF_full_set_WN['decoding_accuracy'].loc[
                                        DF_full_set_WN['MdlType']=='TimeFeats'], 
                            paired=True)
ttest_FreqTime_WN.round(3)

ttest_FreqTime_VS_WN = pg.ttest(DF_full_set_WN['decoding_accuracy'].loc[
    (DF_full_set_WN['MdlType']=='FreqBands') & (DF_full_set_WN['ExpCond']=='VS')], 
                            DF_full_set_WN['decoding_accuracy'].loc[
    (DF_full_set_WN['MdlType']=='TimeFeats') & (DF_full_set_WN['ExpCond']=='VS')], 
                            paired=True)

ttest_FreqTime_VS_WN.round(3)


# 2a: between 
DF_full_set_BW = DF_fullsample_BW.loc[DF_fullsample_BW['feature']=='aggregate', :]
aov_Mdl_BW = pg.rm_anova(data=DF_full_set_BW, dv='decoding_accuracy', 
                              within=['MdlType', 'ExpCond'], subject='subjID')        
aov_Mdl_BW.round(3)    

ttest_FreqTime_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
                                DF_full_set_BW['MdlType']=='FreqBands'], 
                            DF_full_set_BW['decoding_accuracy'].loc[
                                DF_full_set_BW['MdlType']=='TimeFeats'], 
                            paired=True)
ttest_FreqTime_BW.round(3)

ttest_FFTtime_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
                                DF_full_set_BW['MdlType']=='FullFFT'], 
                            DF_full_set_BW['decoding_accuracy'].loc[
                                DF_full_set_BW['MdlType']=='TimeFeats'], 
                            paired=True)
ttest_FFTtime_BW.round(3)



ttest_FFTtime_VS_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
    (DF_full_set_BW['MdlType']=='FreqBands') & (DF_full_set_BW['ExpCond']=='VS')], 
                            DF_full_set_WN['decoding_accuracy'].loc[
    (DF_full_set_BW['MdlType']=='TimeFeats') & (DF_full_set_BW['ExpCond']=='VS')], 
                            paired=True)

ttest_FFTtime_VS_BW.round(3)


# subplot within
plt.figure()
plt.subplot(121)
ax = sns.barplot(data=DF_full_set_WN.round(3), y='decoding_accuracy', x='MdlType',
                 hue='ExpCond', palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")

plt.ylim((.5, 1))
plt.title('Features type\n within subject accuracy')
plt.tight_layout()
plt.legend([],[], frameon=False)

# subplot within
plt.subplot(122)
ax = sns.barplot(data=DF_full_set_BW, y='decoding_accuracy', x='MdlType',
            hue='ExpCond', palette="ch:start=.2,rot=-.3, dark=.4")
plt.ylim((.5, 1))

plt.title('Features type\n between subjects (LOO) accuracy')
plt.tight_layout()


#%% explore top 10 single features in every subcondition

list_DFs_BW_WN = [DF_fullsample_BW, DF_fullsample_WN] 
CompTypes = ['between', 'within']
ExpConds = ['ECEO', 'VS'] # repeated for code readability
MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats']

acc_comp = -1
for iComp in list_DFs_BW_WN:
    
    acc_comp += 1
    this_Comp_str = CompTypes[acc_comp]
    
    for iCond in ExpConds:
        
        tmp_DF = iComp.loc[iComp['ExpCond']==iCond, :]
            
        # bind two columns (feature and MdlType) in order to allow comparisons
        # between single features across all models, in each comparison of interest
        cols_bind = [ 'MdlType', 'feature']
        swap_tmp = tmp_DF.copy()
        swap_tmp['combined'] = tmp_DF[cols_bind].apply(
            lambda row: '\n'.join(row.values.astype(str)), axis=1)
        
        # long to wide
        full_feats_wide = pd.pivot(swap_tmp, index='subjID', 
                                   columns='combined', values='decoding_accuracy')    
        new_idx = full_feats_wide.mean().sort_values(ascending=False)
        full_feats_sorted = full_feats_wide.reindex(new_idx.index, axis=1)

        best10_feats_sorted = full_feats_sorted.iloc[:, 0:10]

        plt.figure()

        ax = sns.stripplot(data=best10_feats_sorted, orient='h', 
                           palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
        ax = sns.barplot(data=best10_feats_sorted, orient='h', errcolor=(.3, .3, .3, 1),
            linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))

        ax.set_yticklabels(best10_feats_sorted.columns, size=8)
        plt.tight_layout()

        ttl_string = iCond + ' ' + this_Comp_str
        plt.title(ttl_string)

        plt.xlabel('balanced accuracy')
        plt.ylabel('model / feature')
        plt.xlim((.6, 1))
        
        cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=len(new_idx))
        acc_ypos = .15; acc_color = 0
        for itxt in new_idx[0:10]:
            
            plt.text(.6, acc_ypos, str(round(itxt, 3)), color=cpal[acc_color])
            acc_ypos += 1; acc_color += 1



# #%% 3. plot the best feats within subjects and bands

# # usual long to wide
# best_feats_wide = pd.pivot(allsubjs_and_bands_best, index='SubjID', 
#                            columns='combined', values='accuracy')    
# new_idx = best_feats_wide.mean().sort_values(ascending=False)
# best_feats_sorted = best_feats_wide.reindex(new_idx.index, axis=1)

# plt.figure()

# ax = sns.stripplot(data=best_feats_sorted, orient='h', 
#                    palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
# ax = sns.barplot(data=best_feats_sorted, orient='h', errcolor=(.3, .3, .3, 1),
#     linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))

# ax.set_yticklabels(best_feats_sorted.columns, size=8)


# plt.tight_layout()
# plt.title('Winning features, (within subjects)')
# plt.xlabel('balanced accuracy')
# plt.ylabel('freq band / feature')
# plt.xlim((.6, 1))

# cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=len(new_idx))
# acc_ypos = .15; acc_color = 0
# for itxt in new_idx:
    
#     plt.text(.6, acc_ypos, str(round(itxt, 3)), color=cpal[acc_color])
#     acc_ypos += 1; acc_color += 1



# plt.show()
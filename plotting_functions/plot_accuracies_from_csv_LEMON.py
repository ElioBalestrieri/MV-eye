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
import os, fnmatch

infold = '../STRG_decoding_accuracy/LEMON/'
plt.close('all')

#%% single subjects

MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats', 'FTM'] #, 'FB_cherrypicked', 'cherrypicked']

filt_fnames = fnmatch.filter(os.listdir(infold), '*' + MdlTypes[0] +'*')
acc = 0
for iname in filt_fnames: 
    iname = iname[0:10]
    filt_fnames[acc] = iname
    acc +=1


allsubjs_within_DFs, allsubjs_between_DFs = [], []

for isubj in filt_fnames:
        
    for iMdl in MdlTypes:
    
        subjname = isubj
                    
        # within subjects            
        fname_within = infold + subjname + '_' + iMdl + '_WS.csv'
        DF_WN = pd.read_csv(fname_within)
        
        tmp = DF_WN.set_index('Unnamed: 0').stack().to_frame().\
                    reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                  'level_1': 'feature', 
                                                  0: 'decoding_accuracy'})         
        tmp = tmp.drop('freq_band', axis=1)
        
        # add columns
        tmp['subjID'] = [subjname] * tmp.shape[0]
        tmp['MdlType'] = [iMdl] * tmp.shape[0]

        allsubjs_within_DFs.append(tmp)

        # # between subjects
        # fname_between = infold + 'ID_' + subjname + '_leftout_' + iExp + '_' + iMdl + '_intersubjs_accs.csv'
        # DF_BW = pd.read_csv(fname_between)

        # tmp = DF_BW.set_index('Unnamed: 0').stack().to_frame().\
        #             reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
        #                                           'level_1': 'feature', 
        #                                           0: 'decoding_accuracy'})   
        # tmp = tmp.drop('freq_band', axis=1)
        
        # # add columns
        # tmp['subjID'] = [subjname] * tmp.shape[0]
        # tmp['ExpCond'] = [iExp] * tmp.shape[0]
        # tmp['MdlType'] = [iMdl] * tmp.shape[0]

        # allsubjs_between_DFs.append(tmp)
             
DF_fullsample_WN = pd.concat(allsubjs_within_DFs, ignore_index=True) 
# DF_fullsample_BW = pd.concat(allsubjs_between_DFs, ignore_index=True) 


#%% filter out subject for which there was a convergence fail.

summary_subjcount = DF_fullsample_WN.groupby('subjID').count()
bad_apples = summary_subjcount['feature']<53
bad_apples = list(bad_apples[bad_apples].index)

idxd_dat = DF_fullsample_WN.set_index('subjID')
idxd_dat = idxd_dat.drop(bad_apples)

DF_fullsample_WN = idxd_dat

#%%

dict_res = {}
for iMdl in MdlTypes:
    
    mask1 = DF_fullsample_WN['MdlType'] == iMdl
    mask2 = DF_fullsample_WN['feature'] == 'full_set'
    
    acc = DF_fullsample_WN['decoding_accuracy'].loc[mask1 & mask2]
    
    dict_res.update({iMdl : acc.mean()})


print(dict_res)

# #%% plotting

# # 2. for each freq band plot the ordered full set of features.
# # moreover, select the best 3 features in each freq band

# temp_MdlTypes = ['TimeFeats', 'FreqBands']# plt.figure()
# # ax = sns.barplot(data=DF_across.round(3), x='MdlType',
# #                  hue='Index', palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")

# DS_WN_BW_list = [DF_fullsample_WN.round(3), DF_fullsample_BW.round(3)]
# list_compname = ['within subj', 'bewteen subj']


# acc_data = -1

# for rounded_DF in DS_WN_BW_list:

#     acc_data += 1
#     name_comp = list_compname[acc_data]

#     for iCond in ExpConds:

#         for iMdl in temp_MdlTypes:

#             # plt.figure()
#             this_DF = rounded_DF.loc[rounded_DF['MdlType']==iMdl, :]
#             this_DF = this_DF.loc[this_DF['ExpCond']==iCond]

#             # go to wide format to allow sorting
#             wide_DF = pd.pivot(this_DF, index='subjID', columns='feature',
#                                values='decoding_accuracy')
#             new_idx = wide_DF.mean().sort_values(ascending=False)
#             wide_DF = wide_DF.reindex(new_idx.index, axis=1)

#             # # plot
#             # ax = sns.barplot(data=wide_DF, orient='h',
#             #                  palette="ch:start=.2,rot=-.3, dark=.4")
#             #
#             # plt.tight_layout()
#             # plt.title(name_comp + ' SVM,\n' + iCond + ', ' +  iMdl)
#             # plt.xlabel('balanced accuracy')
#             # plt.xlim((.5, 1))

#             # plt.show()
    
#%% stats & plots
# 1. compare the full set of features in 3 different conditions: fullFFT, FreqBands
# TimeFeats

# 1. winning "model": fullFFT, vs time feats, vs freq bands

# 1a: within subjects accuracy
DF_full_set_WN = DF_fullsample_WN.loc[DF_fullsample_WN['feature']=='full_set', :]
DF_full_set_WN = DF_full_set_WN.reset_index()
aov_Mdl_WN = pg.rm_anova(data=DF_full_set_WN, dv='decoding_accuracy', 
                              within=['MdlType'], subject='subjID',
                              detailed=False)        
print('\nANOVA within (full)')
print(aov_Mdl_WN.round(3).to_string())
            

mask_ = (DF_fullsample_WN['feature']=='full_set') & ((DF_fullsample_WN['MdlType']=='FreqBands') | (DF_fullsample_WN['MdlType']=='TimeFeats'))
restricted_WN = DF_fullsample_WN.loc[mask_, :].reset_index()

aov_Mdl_WN_restr = pg.rm_anova(data=restricted_WN, dv='decoding_accuracy', 
                              within=['MdlType'], subject='subjID',
                              detailed=False)
print('\nANOVA within (FreqBands & TimeFeats only)')
print(aov_Mdl_WN_restr.round(3).to_string())


ttest_FreqTime_WN = pg.ttest(DF_full_set_WN['decoding_accuracy'].loc[
                                        DF_full_set_WN['MdlType']=='FreqBands'],
                            DF_full_set_WN['decoding_accuracy'].loc[
                                        DF_full_set_WN['MdlType']=='TimeFeats'], 
                            paired=True)
print('\nTTest within FreqBands vs TimeFeats')
print(ttest_FreqTime_WN.round(3).to_string())


# ttest_FreqTime_VS_WN = pg.ttest(DF_full_set_WN['decoding_accuracy'].loc[
#     (DF_full_set_WN['MdlType']=='FreqBands') & (DF_full_set_WN['ExpCond']=='VS')],
#                             DF_full_set_WN['decoding_accuracy'].loc[
#     (DF_full_set_WN['MdlType']=='TimeFeats') & (DF_full_set_WN['ExpCond']=='VS')], 
#                             paired=True)
# print('\nTTest within FreqBands vs TimeFeats in Visual Stimulation')
# print(ttest_FreqTime_VS_WN.round(3).to_string())


# ttest_FreqTime_VS_WN = pg.ttest(DF_fullsample_WN['decoding_accuracy'].loc[
#     (DF_fullsample_WN['MdlType']=='FreqBands') & (DF_fullsample_WN['ExpCond']=='ECEO') & (DF_fullsample_WN['feature']=='full_set')],
#                             DF_fullsample_WN['decoding_accuracy'].loc[
#     (DF_fullsample_WN['MdlType']=='TimeFeats') & (DF_fullsample_WN['ExpCond']=='ECEO') & (DF_fullsample_WN['feature']=='MCL')],
#                             paired=True)
# print('\nTTest within FreqBands vs MCL in ECEO')
# print(ttest_FreqTime_VS_WN.round(3).to_string())


# ttest_FreqTime_ECEO_WN = pg.ttest(DF_full_set_WN['decoding_accuracy'].loc[
#     (DF_full_set_WN['MdlType']=='FreqBands') & (DF_full_set_WN['ExpCond']=='ECEO')],
#                             DF_full_set_WN['decoding_accuracy'].loc[
#     (DF_full_set_WN['MdlType']=='TimeFeats') & (DF_full_set_WN['ExpCond']=='ECEO')], 
#                             paired=True)
# print('\nTTest within FreqBands vs TimeFeats in ECEO')
# print(ttest_FreqTime_ECEO_WN.round(3).to_string())



# # 2a: between 
# DF_full_set_BW = DF_fullsample_BW.loc[DF_fullsample_BW['feature']=='aggregate', :]
# aov_Mdl_BW = pg.rm_anova(data=DF_full_set_BW, dv='decoding_accuracy', 
#                               within=['MdlType', 'ExpCond'], subject='subjID')        
# print('\nANOVA between (full)')
# print(aov_Mdl_BW.round(3).to_string())

# ttest_FreqTime_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
#                                 DF_full_set_BW['MdlType']=='FreqBands'], 
#                             DF_full_set_BW['decoding_accuracy'].loc[
#                                 DF_full_set_BW['MdlType']=='TimeFeats'], 
#                             paired=True)
# print('\nTTest between FreqBands vs TimeFeats')
# print(ttest_FreqTime_BW.round(3).to_string())

# ttest_FFTtime_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
#                                 DF_full_set_BW['MdlType']=='FullFFT'], 
#                             DF_full_set_BW['decoding_accuracy'].loc[
#                                 DF_full_set_BW['MdlType']=='TimeFeats'], 
#                             paired=True)
# print('\nTTest between FullFFT vs TimeFeats')
# print(ttest_FFTtime_BW.round(3).to_string())

# ttest_FreqBandstime_VS_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='FreqBands') & (DF_full_set_BW['ExpCond']=='VS')], 
#                             DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='TimeFeats') & (DF_full_set_BW['ExpCond']=='VS')], 
#                             paired=True)
# print('\nTTest between FreqBands vs TimeFeats in Visual Stimulation')
# print(ttest_FreqBandstime_VS_BW.round(3).to_string())


# ttest_FreqsTime_ECEO_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='FreqBands') & (DF_full_set_BW['ExpCond']=='ECEO')], 
#                             DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='TimeFeats') & (DF_full_set_BW['ExpCond']=='ECEO')], 
#                             paired=True)
# print('\nTTest within FreqBands vs TimeFeats in ECEO')
# print(ttest_FreqsTime_ECEO_BW.round(3).to_string())

# ttest_FFTTimeFeats_VS_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='FullFFT') & (DF_full_set_BW['ExpCond']=='VS')],
#                             DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='TimeFeats') & (DF_full_set_BW['ExpCond']=='VS')],
#                             paired=True)
# print('\nTTest between FFT vs TimeFeats in Visual Stimulation')
# print(ttest_FFTTimeFeats_VS_BW.round(3).to_string())

# ttest_FFtTimeFeats_ECEO_BW = pg.ttest(DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='FullFFT') & (DF_full_set_BW['ExpCond']=='ECEO')],
#                             DF_full_set_BW['decoding_accuracy'].loc[
#     (DF_full_set_BW['MdlType']=='TimeFeats') & (DF_full_set_BW['ExpCond']=='ECEO')],
#                             paired=True)
# print('\nTTest between FullFFT vs TimeFeats in ECEO')
# print(ttest_FFtTimeFeats_ECEO_BW.round(3).to_string())


#%% subplot within
plt.figure()
# plt.subplot(221)
ax = sns.barplot(data=DF_full_set_WN.round(3), y='decoding_accuracy', x='MdlType',
                  palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")

plt.ylim((.5, 1))
plt.title('Features type\n within subject accuracy')
plt.legend([],[], frameon=False)

# # subplot between
# plt.subplot(222)
# ax = sns.barplot(data=DF_full_set_BW, y='decoding_accuracy', x='MdlType',
#             hue='ExpCond', palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")
# plt.ylim((.5, 1))
# plt.legend([],[], frameon=False)

# plt.title('Features type\n between subjects (LOO) accuracy')

# # subplot across
# plt.subplot(223)
# ax = sns.barplot(data=DF_across.round(3), x='MdlType', y='decoding accuracy',
#                   hue='ExpCond', palette="ch:start=.2,rot=-.3, dark=.4", errorbar=None)
# plt.ylim((.5, 1))
# plt.title('Features type\n across subjects accuracy')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# plt.tight_layout()
# plt.show()

#%% explore top 10 single features in every subcondition

list_DFs_BW_WN = [DF_fullsample_WN] # DF_fullsample_BW, 
CompTypes = ['between', 'within']
ExpConds = ['ECEO', 'VS'] # repeated for code readability
MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats', 'FreqBands']

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
            linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0),
            errorbar="se")

        ax.set_yticklabels(best10_feats_sorted.columns, size=8)

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

        plt.tight_layout()

        plt.show()


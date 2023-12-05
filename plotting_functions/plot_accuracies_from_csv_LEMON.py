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

#%% single subjects, WS

MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats', 'FTM'] #, 'FB_cherrypicked', 'cherrypicked']

filt_fnames = fnmatch.filter(os.listdir(infold), 'sub-*' + MdlTypes[0] +'*')
acc = 0
for iname in filt_fnames: 
    iname = iname[0:10]
    filt_fnames[acc] = iname
    acc +=1


allsubjs_within_DFs = []

for isubj in filt_fnames:
        
    for iMdl in MdlTypes:
    
        subjname = isubj
                    
        # within subjects            
        fname_within = infold + subjname + '_' + iMdl + '_WS.csv'
        DF_WN = pd.read_csv(fname_within)
        
        tmp = DF_WN.set_index('Unnamed: 0').stack().to_frame().\
                    reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                  'level_1': 'feature', 
                                                  0: 'decoding accuracy'})         
        tmp = tmp.drop('freq_band', axis=1)
        
        # add columns
        tmp['subjID'] = [subjname] * tmp.shape[0]
        tmp['model'] = [iMdl] * tmp.shape[0]

        allsubjs_within_DFs.append(tmp)

             
DF_fullsample_WN = pd.concat(allsubjs_within_DFs, ignore_index=True) 

#%% between subjects

dict_BS = {'model' : [],
           'decoding accuracy' : [],
           'fold' : []}

for ifold in range(5):
    
    for iMdl in MdlTypes:
        
        fname_fold = infold + 'Fold_' + str(ifold) + '_' + iMdl + '_BS_fullset.csv'
        DF_tmp = pd.read_csv(fname_fold)
        
        dict_BS['decoding accuracy'].append(DF_tmp.loc[0, 'test_accuracy'])
        dict_BS['model'].append(iMdl)
        dict_BS['fold'].append('Fold_' + str(ifold))

DF_BS = pd.DataFrame.from_dict(dict_BS)



#%% across subjects

svm_types = ['', 'LinearSVM_']

dict_AS = {'model' : [],
           'decoding accuracy' : [],
           'SVM type': []}

for iSVM in svm_types:

    match iSVM:

        case '':       
            SVMtype = 'GaussianSVM'
            
        case 'LinearSVM_':            
            SVMtype = 'LinearSVM'

    for iMdl in MdlTypes:

        fname_cond = infold + 'AcrossSubjs_PC_' + iSVM + iMdl + '.csv'

        dict_AS['model'].append(iMdl)
        dict_AS['SVM type'].append(SVMtype)
        
        try:            
            DF_tmp = pd.read_csv(fname_cond)
            fileread = True            
        except:
            dec_acc = np.nan
            fileread = False 
                        
        if fileread:
            
            try:            
                dec_acc = DF_tmp.loc[0, 'PC_aggregate']            
            except:
                dec_acc = DF_tmp.loc[0, 'PC_full_set']

        dict_AS['decoding accuracy'].append(dec_acc)   
            

DF_AS = pd.DataFrame.from_dict(dict_AS)

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
    
    mask1 = DF_fullsample_WN['model'] == iMdl
    mask2 = DF_fullsample_WN['feature'] == 'full_set'
    
    acc = DF_fullsample_WN['decoding accuracy'].loc[mask1 & mask2]
    
    dict_res.update({iMdl : acc.mean()})


print(dict_res)
    
#%% stats & plots
# 1. compare the full set of features in 3 different conditions: fullFFT, FreqBands
# TimeFeats

# 1. winning "model": fullFFT, vs time feats, vs freq bands

# 1a: within subjects accuracy
DF_full_set_WN = DF_fullsample_WN.loc[DF_fullsample_WN['feature']=='full_set', :]
DF_full_set_WN = DF_full_set_WN.reset_index()
aov_Mdl_WN = pg.rm_anova(data=DF_full_set_WN, dv='decoding accuracy', 
                              within=['model'], subject='subjID',
                              detailed=False)        
print('\nANOVA within (full)')
print(aov_Mdl_WN.round(3).to_string())
            

mask_ = (DF_fullsample_WN['feature']=='full_set') & ((DF_fullsample_WN['model']=='FreqBands') | (DF_fullsample_WN['model']=='TimeFeats'))
restricted_WN = DF_fullsample_WN.loc[mask_, :].reset_index()

aov_Mdl_WN_restr = pg.rm_anova(data=restricted_WN, dv='decoding accuracy', 
                              within=['model'], subject='subjID',
                              detailed=False)
print('\nANOVA within (FreqBands & TimeFeats only)')
print(aov_Mdl_WN_restr.round(3).to_string())


ttest_FreqTime_WN = pg.ttest(DF_full_set_WN['decoding accuracy'].loc[
                                        DF_full_set_WN['model']=='FreqBands'],
                            DF_full_set_WN['decoding accuracy'].loc[
                                        DF_full_set_WN['model']=='TimeFeats'], 
                            paired=True)
print('\nTTest within FreqBands vs TimeFeats')
print(ttest_FreqTime_WN.round(3).to_string())


ttest_FFTTime_WN = pg.ttest(DF_full_set_WN['decoding accuracy'].loc[
                                        DF_full_set_WN['model']=='FullFFT'],
                            DF_full_set_WN['decoding accuracy'].loc[
                                        DF_full_set_WN['model']=='TimeFeats'], 
                            paired=True)
print('\nTTest within FullFFT vs TimeFeats')
print(ttest_FFTTime_WN.round(3).to_string())

# 2a: between 
aov_Mdl_BW = pg.rm_anova(data=DF_BS, dv='decoding accuracy', 
                              within=['model'], subject='fold')        
print('\nANOVA between (full)')
print(aov_Mdl_BW.round(3).to_string())

ttest_FreqTime_BW = pg.ttest(DF_BS['decoding accuracy'].loc[
                                DF_BS['model']=='FreqBands'], 
                            DF_BS['decoding accuracy'].loc[
                                DF_BS['model']=='TimeFeats'], 
                            paired=True)
print('\nTTest between FreqBands vs TimeFeats')
print(ttest_FreqTime_BW.round(3).to_string())

ttest_FFTtime_BW = pg.ttest(DF_BS['decoding accuracy'].loc[
                                DF_BS['model']=='FullFFT'], 
                            DF_BS['decoding accuracy'].loc[
                                DF_BS['model']=='TimeFeats'], 
                            paired=True)
print('\nTTest between FullFFT vs TimeFeats')
print(ttest_FFTtime_BW.round(3).to_string())


#%% subplot within
plt.figure()
# ax = sns.barplot(data=DF_full_set_WN.round(3), y='decoding accuracy', x='model',
#                   palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")

plt.subplot(131)

vioplinpltdata_WN = DF_full_set_WN.loc[DF_full_set_WN['model']!='FTM', :] 

ax = sns.stripplot(data=vioplinpltdata_WN, y='decoding accuracy', x='model',
                  palette="ch:start=.2,rot=-.3, dark=.4", size=2)

ax = sns.violinplot(data=vioplinpltdata_WN , y='decoding accuracy', x='model',
                  palette="ch:start=.2,rot=-.3, dark=.4", fill=False)

# ax = sns.stripplot(data=DF_full_set_WN.round(3), y='decoding accuracy', x='model',
#                   palette="ch:start=.2,rot=-.3, dark=.4", size=1)


plt.ylim((.96, 1.001))
plt.title('Within subject accuracy \nGaussian SVM')
# plt.legend([],[], frameon=False)

# subplot between
plt.subplot(132)

# ax = sns.barplot(data=DF_BS, y='decoding accuracy', x='model',
#                  palette="ch:start=.2,rot=-.3, dark=.4", errorbar="se")

# ax = sns.stripplot(data=DF_BS.loc[DF_BS['model']!='FTM', :], y='decoding accuracy', x='model',
#                   palette="ch:start=.2,rot=-.3, dark=.4", size=3)

ax = sns.violinplot(data=DF_BS.loc[DF_BS['model']!='FTM', :], y='decoding accuracy', x='model',
                 palette="ch:start=.2,rot=-.3, dark=.4", fill=False)


plt.ylim((.74, .85))
plt.legend([],[], frameon=False)
plt.title('5-fold Between subjects crossval \nLinear SVM')



# subplot across
plt.subplot(133)
lin_DF_AS = DF_AS.loc[(DF_AS['SVM type']=='LinearSVM') & (DF_AS['model']!='FTM'), :]
ax = sns.barplot(data=lin_DF_AS, x='model', y='decoding accuracy',
                 palette="ch:start=.2,rot=-.3, dark=.4", errorbar=None)
plt.ylim((.7, .85))
plt.title('Across subjects accuracy \nLinear SVM')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.suptitle('LEMON Classification results')

plt.tight_layout()

#%% explore top 10 single features in every subcondition
    
# bind two columns (feature and MdlType) in order to allow comparisons
# between single features across all models, in each comparison of interest
cols_bind = [ 'model', 'feature']
swap_tmp = DF_fullsample_WN.copy()
swap_tmp['combined'] = DF_fullsample_WN[cols_bind].apply(
    lambda row: '\n'.join(row.values.astype(str)), axis=1)
        
swap_tmp['subjID'] = swap_tmp.index

# long to wide
full_feats_wide = pd.pivot(swap_tmp, index='subjID', 
                            columns='combined', values='decoding accuracy')    
new_idx = full_feats_wide.mean().sort_values(ascending=False)
full_feats_sorted = full_feats_wide.reindex(new_idx.index, axis=1)

best20_feats_sorted = full_feats_sorted.iloc[:, 0:20]

plt.figure()

ax = sns.barplot(data=best20_feats_sorted, orient='h', errcolor=(.3, .3, .3, 1),
    linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0),
    errorbar="se")

ax.set_yticklabels(best20_feats_sorted.columns, size=8)

plt.title('LEMON within subjects\n20 best features')

plt.xlabel('balanced accuracy')
plt.ylabel('model / feature')
plt.xlim((.6, 1))

cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=len(new_idx))
acc_ypos = .15; acc_color = 0
for itxt in new_idx[0:20]:
    
    plt.text(.6, acc_ypos, str(round(itxt, 3)), color=cpal[acc_color])
    acc_ypos += 1; acc_color += 1

plt.tight_layout()

plt.show()


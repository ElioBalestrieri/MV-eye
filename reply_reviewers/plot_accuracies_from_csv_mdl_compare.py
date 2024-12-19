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

infold = '../STRG_decoding_accuracy/rev_reply_MEG/'
infold_old = '../STRG_decoding_accuracy/Mdl_comparison/'
# plt.close('all')

#%% single subjects

ExpConds = ['ECEO', 'VS'] 
MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats']# , 'FTM'] #, 'FB_cherrypicked', 'cherrypicked']

up_to_subj = 29

allsubjs_within_DFs, allsubjs_between_DFs = [], []

for isubj in range(up_to_subj):

    for iExp in ExpConds:
        
        for iMdl in MdlTypes:
    
            subjname = f'{isubj+1:02d}'
                    
            # within subjects          
            if iMdl == 'TimeFeats':
                fname_within = infold + subjname + '_' + iExp + '_' + iMdl + '_customFOOOF.csv'
            else:
                fname_within = infold + subjname + '_' + iExp + '_' + iMdl + '.csv'
                

            DF_WN = pd.read_csv(fname_within)
            
            tmp = DF_WN.stack().to_frame().\
                        reset_index().rename(columns={'level_0': 'freq_band', 
                                                      'level_1': 'feature', 
                                                      0: 'decoding_accuracy'})         
            tmp = tmp.drop('freq_band', axis=1)
            
            # add columns
            tmp['subjID'] = [subjname] * tmp.shape[0]
            tmp['ExpCond'] = [iExp] * tmp.shape[0]
            tmp['MdlType'] = [iMdl] * tmp.shape[0]

            allsubjs_within_DFs.append(tmp)

            # between subjects
            fname_between = infold_old + 'ID_' + subjname + '_leftout_' + iExp + '_' + iMdl + '_intersubjs_accs.csv'
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


#%% across subjects comparisons

#
DF = pd.read_csv(infold + 'AS_scaled_WS.csv')
full_sets_mask = DF.feature.str.contains('full')
DFfull = DF.loc[full_sets_mask, :]

SVM_DFfull = DFfull.loc[DFfull['classifier']=='SVM', :]
SVM_DFfull = SVM_DFfull.sort_values(by='condition', ascending=False)
SVM_DFfull = SVM_DFfull.sort_values(by='model')

aov_Mdl_across = pg.rm_anova(data=SVM_DFfull, dv='accuracy', 
                             within=['model', 'condition'], subject='cvFolds',
                             detailed=True)        
print(aov_Mdl_across)

pw_comps = pg.pairwise_tests(data=SVM_DFfull, dv='accuracy', 
                             within=['condition', 'model'], subject='cvFolds', 
                             effsize='cohen', padjust='fdr_bh')

pw_comps.to_csv('pairwise_across.csv')
print(pw_comps.round(3).to_string())

#%% within subjects stats
DF_full_set_WN = DF_fullsample_WN.loc[DF_fullsample_WN['feature']=='full_set', :]
aov_Mdl_WN = pg.rm_anova(data=DF_full_set_WN, dv='decoding_accuracy', 
                              within=['MdlType', 'ExpCond'], subject='subjID',
                              detailed=True)        
print('\nANOVA within (full)')
print(aov_Mdl_WN.round(3).to_string())
DF_full_set_WN.groupby(['ExpCond', 'MdlType'])['decoding_accuracy'].mean()   

pw_tests = pg.pairwise_tests(data=DF_full_set_WN, dv='decoding_accuracy',
                             within=['ExpCond', 'MdlType'],  subject='subjID',
                             alternative='two-sided', interaction=True, 
                             padjust='fdr_bh', effsize='cohen')

print(pw_tests.round(3).to_string())


#%% between subjects stats
DF_full_set_BW = DF_fullsample_BW.loc[DF_fullsample_BW['feature']=='aggregate', :]
aov_Mdl_BW = pg.rm_anova(data=DF_full_set_BW, dv='decoding_accuracy', 
                              within=['MdlType', 'ExpCond'], subject='subjID')        
print('\nANOVA between (full)')
print(aov_Mdl_BW.round(3).to_string())
DF_full_set_BW.groupby(['ExpCond', 'MdlType'])['decoding_accuracy'].mean()   


pw_tests = pg.pairwise_tests(data=DF_full_set_BW, dv='decoding_accuracy',
                             within=['ExpCond', 'MdlType'],  subject='subjID',
                             alternative='two-sided', interaction=True, 
                             padjust='fdr_bh', effsize='cohen')

print(pw_tests.round(3).to_string())


#%% subplot within
plt.figure()
plt.subplot(131)
plotdata1 = DF_full_set_WN

ax = sns.violinplot(data=plotdata1, y='decoding_accuracy', x='MdlType',
                  palette="ch:start=.2,rot=-.3, dark=.4", fill=False,
                  split=True, hue='ExpCond', cut=0, order=MdlTypes)

plt.ylim((.5, 1.001))
plt.title('Within subjects accuracy', fontsize=16)
plt.legend([],[], frameon=False)
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('Feature set', fontsize=12)

# subplot between
plt.subplot(132)

plotdata2 = DF_full_set_BW.loc[DF_full_set_BW['MdlType']!='FTM', :]
ax = sns.violinplot(data=plotdata2, y='decoding_accuracy', x='MdlType',
                  palette="ch:start=.2,rot=-.3, dark=.4", fill=False,
                  split=True, hue='ExpCond', cut=0)
plt.ylabel('balanced accuracy', fontsize=12)
plt.xlabel('Feature set', fontsize=12)

plt.ylim((.3, 1.001))

plt.title('Between subjects accuracy', fontsize=16)
plt.legend([],[], frameon=False)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# plt.tight_layout()


# subplot across
plt.subplot(133)

sns.violinplot(data=SVM_DFfull, y='accuracy', x='model',
                  palette="ch:start=.2,rot=-.3, dark=.4", fill=False,
                  split=True, hue='condition', cut=0, order=MdlTypes)

plt.title('Across subjects accuracy', fontsize=16)
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('Feature set', fontsize=12)


plt.tight_layout()




#%%


ax = sns.barplot(data=plotdata3, x='model', y='accuracy',
                  hue='condition', palette="ch:start=.2,rot=-.3, dark=.4", errorbar=None)
plt.ylim((.75, 1))
plt.title('Across subjects accuracy', fontsize=16)
plt.legend(title='Classification type', labels=['EC / EO', 'BSL / VS'])
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.ylabel('balanced accuracy', fontsize=12)
plt.xlabel('Feature set', fontsize=12)
plt.tight_layout()
# plt.show()

#%% explore top 10 single features in every subcondition

list_DFs_BW_WN = [DF_fullsample_BW, DF_fullsample_WN] 
CompTypes = ['between', 'within']
ExpConds = ['ECEO', 'VS'] # repeated for code readability
MdlTypes = ['FullFFT', 'FreqBands', 'TimeFeats']

# plt.figure()

acc_comp = -1

acc_fig = 0
for iComp in list_DFs_BW_WN:
    
    acc_comp += 1
    this_Comp_str = CompTypes[acc_comp]
    
    for iCond in ExpConds:
        
        tmp_DF = iComp.loc[(iComp['ExpCond']==iCond) & (iComp['feature']!='fullFFT'), :]
            
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
        # acc_fig +=1
        # plt.subplot(2, 2, acc_fig)

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

# 
    
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
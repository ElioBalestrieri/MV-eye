# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os
import dask
from datetime import datetime
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test

#%% folder(s) definition & cond type

# output folder
infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/Mdl_comparison/'
outfold = infold

if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% repeat concatenation to get subjects labels of single trials

ExpConds = ['ECEO', 'VS']
mdltypes = ['TimeFeats', 'FreqBands']

back_infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/Mdl_comparison/'


# concatenate files between participant, after within-participant normalization
list_accs = [];   
L_Xtrain, L_Xtest, PC_IDs = [], [], []
for imdl in mdltypes:
        
    list_ExpConds = []  
    for ThisExpCond in ExpConds:                
        list_ExpConds.append(ThisExpCond + '_' + imdl)

    fullX_train, fullX_test, Y_train, Y_test, subjID_trials_labels = cat_subjs_train_test(back_infold, strtsubj=0, endsubj=29, 
                                                                    ftype=list_ExpConds, tanh_flag=True, 
                                                                    compress_flag=True, pca_kept_var=.9)     
    L_Xtrain.append(fullX_train['PC_full_set'])
    L_Xtest.append(fullX_test['PC_full_set'])
    PC_IDs.append(fullX_train['list_PC_identifiers'])
    
mergedMdls_train = np.concatenate(L_Xtrain, axis=1)
mergedMdls_test = np.concatenate(L_Xtest, axis=1)
merged_IDs = np.array(PC_IDs[0] + PC_IDs[1]) 


#%% Pipeline object definition

# functions for factor analysis
from sklearn.decomposition import FactorAnalysis

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# # pipeline definer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import balanced_accuracy_score

#%% load files

acc_expCond = 0; 
FA_mdls_list, LDA_mdls_list  = [], []


fnameTrainX = 'SeqFeatSel_condscollapsed_trainX.csv'
trainX = pd.read_csv(infold+fnameTrainX)

fnameTestX = 'SeqFeatSel_condscollapsed_testX.csv'
testX = pd.read_csv(infold+fnameTestX)

fnameTrainY = 'SeqFeatSel_condscollapsed_trainY.csv'
trainY = pd.read_csv(infold+fnameTrainY)

fnameTestY = 'SeqFeatSel_condscollapsed_testY.csv'
testY = pd.read_csv(infold+fnameTestY)

#%% data selection & feature relabeling

# select the full (aggregate) data for all the analyses that follow    
X_full = pd.concat([trainX, testX])
Y_full = pd.concat([trainY, testY])

Y_full = Y_full.drop(['Unnamed: 0'], axis=1)
X_full = X_full.drop(['Unnamed: 0'], axis=1)

#%% apply factor analysis

transformer_FA = FactorAnalysis(n_components=3, random_state=42)
X_transformed_FA = transformer_FA.fit_transform(X_full)

FA_mdls_list.append(transformer_FA)    

#%% apply LDA

this_LDA = LinearDiscriminantAnalysis(n_components=3)
this_LDA.fit(X_full, Y_full['label'])

X_transformed_LDA = this_LDA.transform(X_full)

LDA_mdls_list.append(this_LDA)

#%% group & plot loadings according to feature categories -average across parcels
feats_unique_labels = X_full.columns
array_loads = np.empty((2, len(feats_unique_labels)))
acc_feat = 0
for ifeat in feats_unique_labels:

    mask_ = [s == ifeat for s in feats_unique_labels]
    array_loads[0, acc_feat] = this_LDA.coef_[0, mask_].mean()
    array_loads[1, acc_feat] = transformer_FA.components_[0, mask_].mean()

    acc_feat += 1

DF_loadings = pd.DataFrame(array_loads.T, columns=['LDA loads', 'FA loads'], 
                           index=feats_unique_labels)


DF_loadings.plot.barh()
plt.tight_layout()

#%% find 2 comps with highest absolute LDA scores for 2d representation
DFrstrd = DF_loadings.reindex(DF_loadings['LDA loads'].abs().sort_values().index)

best2feats = DFrstrd.index[[-1, -2]]


#%% represent data in 2D based on FA
tmp = {'LatentFactor1' : X_transformed_FA[:,0],
       'LatentFactor2' : X_transformed_FA[:,1],
       'LatentFactor3' : X_transformed_FA[:,2],       
       'LDA_comp1' : X_transformed_LDA[:,0],
       'LDA_comp2' : X_transformed_LDA[:,1],
       'LDA_comp3' : X_transformed_LDA[:,2],       
       'condition' : np.array(Y_full['label']),
       DFrstrd.index[-1] : np.array(X_full[DFrstrd.index[-1]]),
       DFrstrd.index[-2] : np.array(X_full[DFrstrd.index[-2]])}

DF_transformations = pd.DataFrame.from_dict(data=tmp)

ExpCondsLabels=['EC', 'EO', 'bsl', 'VS']

unique_subj_ids = list(set(subjID_trials_labels))

allsubjs_4conds_FA, allsubjs_4conds_LDA = [], []
for inum in range(4):
    
    thisnum=inum+1
    lgcl_cond = np.array(DF_transformations['condition']==thisnum)
    
    allsubjs_FA, allsubjs_LDA = [], []
    for isubj in unique_subj_ids:
        
        lgcl_subj = np.array(subjID_trials_labels) == isubj
        mask_ = lgcl_subj & lgcl_cond
        
        x_FA = DF_transformations['LatentFactor1'].loc[mask_]
        y_FA = DF_transformations['LatentFactor2'].loc[mask_]
        z_FA = DF_transformations['LatentFactor3'].loc[mask_]
    
        allsubjs_FA.append([x_FA.mean(), y_FA.mean(), z_FA.mean()])

        x_LDA = DF_transformations['LDA_comp1'].loc[mask_]
        y_LDA = DF_transformations['LDA_comp2'].loc[mask_]
        z_LDA = DF_transformations['LDA_comp3'].loc[mask_]
        
        allsubjs_LDA.append([x_LDA.mean(), y_LDA.mean(), z_LDA.mean()])


    allsubjs_FA = np.array(allsubjs_FA); allsubjs_LDA = np.array(allsubjs_LDA)
    allsubjs_4conds_FA.append(allsubjs_FA)
    allsubjs_4conds_LDA.append(allsubjs_LDA)
            
allsubjs_4conds_FA = np.array(allsubjs_4conds_FA)      
allsubjs_4conds_LDA = np.array(allsubjs_4conds_LDA)      


#%% represent subjects in 3d space

fig = plt.figure()
ax1 = plt.axes(projection='3d')

for icond in range(4):
    
    ax1.scatter3D(allsubjs_4conds_FA[icond, :, 0], 
                  allsubjs_4conds_FA[icond, :, 1],
                  allsubjs_4conds_FA[icond, :, 2])
    
for isubj in range(29):

    ax1.plot3D(allsubjs_4conds_FA[:, isubj, 0], 
                allsubjs_4conds_FA[:, isubj, 1],
                allsubjs_4conds_FA[:, isubj, 2], 'gray', alpha=.6, linewidth=.1)


ax1.legend(ExpCondsLabels)


#%%


fig = plt.figure()
ax2 = plt.axes(projection='3d')


for icond in range(4):
    
    ax2.scatter3D(allsubjs_4conds_LDA[icond, :, 0], 
                  allsubjs_4conds_LDA[icond, :, 1],
                  allsubjs_4conds_LDA[icond, :, 2])
    
for isubj in range(29):

    ax2.plot3D(allsubjs_4conds_LDA[:, isubj, 0], 
               allsubjs_4conds_LDA[:, isubj, 1],
               allsubjs_4conds_LDA[:, isubj, 2], 'gray', alpha=.6, linewidth=.1)

ax2.legend(ExpCondsLabels)



# ax1.scatter3D(x_FA, y_FA, z_FA, s=2, alpha=.5)
# ax2.scatter3D(x_LDA, y_LDA, z_LDA, s=2, alpha=.5)    
# ax2.legend(ExpCondsLabels)
 

#%% 
    
plt.figure()

plt.subplot(222)
sns.kdeplot(data=DF_transformations, x='LatentFactor1', y='LatentFactor2', hue='condition', 
            levels=np.array([0, .5]), alpha=.8)        
plt.subplot(223)
sns.kdeplot(data=DF_transformations, x='LatentFactor2', y='LatentFactor3', hue='condition', 
            levels=np.array([0, .5]), alpha=.8)        
plt.subplot(224)
sns.kdeplot(data=DF_transformations, x='LatentFactor1', y='LatentFactor3', hue='condition', 
            levels=np.array([0, .5]), alpha=.8)        


plt.figure()

plt.subplot(222)
sns.kdeplot(data=DF_transformations, x='LDA_comp1', y='LDA_comp2', hue='condition', 
            levels=np.array([0, .5]), alpha=.8)        
plt.subplot(223)
sns.kdeplot(data=DF_transformations, x='LDA_comp2', y='LDA_comp3', hue='condition', 
            levels=np.array([0, .5]), alpha=.8)        
plt.subplot(224)
sns.kdeplot(data=DF_transformations, x='LDA_comp1', y='LDA_comp3', hue='condition', 
            levels=np.array([0, .5]), alpha=.8)        



#     #%% get single subject transformation on 2d plane
    
#     condtypes = [0, 1]; list_SUBJID = list(set(allsubjs_ID))
    
#     all_subjs_dict = {'LatentFactor1' : [],
#                       'LatentFactor2' : [],
#                       'LDA_transformed' : [],
#                       'condition' : []}
    
#     for SUBJid in list_SUBJID:
        
#         # get logical for the current subject    
#         lgcl_this_subj = [s == SUBJid for s in allsubjs_ID]
    
#         # this_subj_dict = {'comp1' : [],
#         #                   'comp2' : [],
#         #                   'condition' : []}
    
#         for icond in range(2):
            
#             lgcl_this_cond = allsubjs_Y == (icond+1)
    
#             mask_ = lgcl_this_cond & lgcl_this_subj
            
#             this_transition_FA = X_transformed_FA[mask_, :]
#             this_transition_LDA = X_transformed_LDA[mask_, 0]
    
#             # this_subj_dict['comp1'].extend(list(this_transition[:, 0]))        
#             # this_subj_dict['comp2'].extend(list(this_transition[:, 1]))        
#             # this_subj_dict['condition'].extend([condnames[icond]] * sum(mask_))        
    
#             all_subjs_dict['LatentFactor1'].append(this_transition_FA[:, 0].mean())        
#             all_subjs_dict['LatentFactor2'].append(this_transition_FA[:, 1].mean())     
#             all_subjs_dict['LDA_transformed'].append(this_transition_LDA.mean())     
            
#             all_subjs_dict['condition'].append(condnames[icond])        
    
    
    
#         # DF_this_subj = pd.DataFrame.from_dict(this_subj_dict)
#         # plt.figure()
#         # sns.kdeplot(data=DF_this_subj, x='comp1', y='comp2', hue='condition')
    
    
#     DF_all_subjs = pd.DataFrame.from_dict(all_subjs_dict)
#     plt.figure()
#     plt.subplot(221)
#     sns.kdeplot(data=DF_all_subjs, x='LatentFactor1', y='LatentFactor2', hue='condition')
#     plt.tight_layout()
    
#     plt.subplot(222)
#     plt.scatter(DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[0]], 
#                 DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[0]], s=20)
    
#     plt.scatter(DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[1]], 
#                 DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[1]], s=20)
    
#     plt.plot(np.array([DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[0]], 
#              DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[1]]]), 
#              np.array([DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[0]],
#                       DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[1]]]),
#              'k', linewidth=.5, alpha=.1)
#     plt.xlabel('LatentFactor1')
#     plt.ylabel('LatentFactor2')
#     plt.tight_layout()
    
#     plt.subplot(223)
#     sns.kdeplot(data=DF_all_subjs, x='LDA_transformed', hue='condition')
#     plt.tight_layout()
    
#     plt.suptitle(ThisExpCond)
    
#     acc_expCond +=1
    
    
# plt.figure()
# plt.subplot(121)
# plt.scatter(LDA_mdls_list[0].coef_, LDA_mdls_list[1].coef_, s=10)
# plt.title('coeffs LDA')
# plt.xlabel('VS')
# plt.ylabel('ECEO')
# plt.tight_layout()

# plt.subplot(122)
# plt.scatter(FA_mdls_list[0].components_[0, :], 
#             FA_mdls_list[1].components_[0, :], s=10)
# plt.title('coeffs FA \n(1st factor)')
# plt.xlabel('VS')
# plt.ylabel('ECEO')
# plt.tight_layout()
# #%% 


    
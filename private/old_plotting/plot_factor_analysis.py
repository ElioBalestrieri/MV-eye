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

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs

#%% folder(s) definition & cond type

# output folder
infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/'
outfold = infold

if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# current condition type
# this is provided by input function from the parent bash script

ExpConds = ['VS', 'ECEO']
allCondNames = [['baseline', 'VS'], ['EC', 'EO']]

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

# plt.figure()
for ThisExpCond in ExpConds:

    fnameTrainX = 'SeqFeatSel_' + ThisExpCond + '_trainX.csv'
    trainX = pd.read_csv(infold+fnameTrainX)

    fnameTestX = 'SeqFeatSel_' + ThisExpCond + '_testX.csv'
    testX = pd.read_csv(infold+fnameTestX)

    fnameTrainY = 'SeqFeatSel_' + ThisExpCond + '_trainY.csv'
    trainY = pd.read_csv(infold+fnameTrainY)

    fnameTestY = 'SeqFeatSel_' + ThisExpCond + '_testY.csv'
    testY = pd.read_csv(infold+fnameTestY)

    #%% data selection & feature relabeling

    # select the full (aggregate) data for all the analyses that follow    
    X_full = pd.concat([trainX, testX])
    Y_full = pd.concat([trainY, testY])

    Y_full = Y_full.drop(['Unnamed: 0'], axis=1)
    X_full = X_full.drop(['Unnamed: 0'], axis=1)

    #%% apply factor analysis
    
    transformer_FA = FactorAnalysis(n_components=2, random_state=42)
    X_transformed_FA = transformer_FA.fit_transform(X_full)
    
    FA_mdls_list.append(transformer_FA)    
    
    #%% apply LDA
    
    this_LDA = LinearDiscriminantAnalysis(n_components=1)
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
    
    # plt.subplot(121)
    # plt.figure()
    DF_loadings.plot.barh()
    plt.tight_layout()
    plt.title(ThisExpCond)
    
    #%% find 2 comps with highest absolute LDA scores for 2d representation
    DFrstrd = DF_loadings.reindex(DF_loadings['LDA loads'].abs().sort_values().index)
    
    best2feats = DFrstrd.index[[-1, -2]]
    
    
    #%% represent data in 2D based on FA
    tmp = {'LatentFactor1' : X_transformed_FA[:,0],
           'LatentFactor2' : X_transformed_FA[:,1],
           'LDA_transformed' : X_transformed_LDA[:,0],
           'condition' : np.array(Y_full['label']),
           DFrstrd.index[-1] : np.array(X_full[DFrstrd.index[-1]]),
           DFrstrd.index[-2] : np.array(X_full[DFrstrd.index[-2]])}
    
    DF_transformations = pd.DataFrame.from_dict(data=tmp)
    plt.figure(); plt.subplot(121)
    sns.kdeplot(data=DF_transformations, x='LatentFactor1', y='LatentFactor2', hue='condition')
        
    plt.subplot(122)
    sns.kdeplot(data=DF_transformations, x=DFrstrd.index[-1], y=DFrstrd.index[-2], hue='condition')
    
        
    
    
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


    
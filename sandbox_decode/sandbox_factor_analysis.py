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

#input folder 
infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/'

# output folder
outfold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# current condition type
# this is provided by input function from the parent bash script

ExpConds = ['VS', 'ECEO']
allCondNames = [['baseline', 'VS'], ['EC', 'EO']]
mdltype = 'TimeFeats' # 'FreqBandsSimple', 'FreqBands', 'FullFFT', 'TimeFeats'

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
for ThisExpCond in ExpConds:

    condnames = allCondNames[acc_expCond]    
                    
    fname_X = infold + ThisExpCond + '_allsubjs_X_'  + mdltype + '.pickle'
    fname_Y = infold + ThisExpCond + '_allsubjs_Y_'  + mdltype + '.pickle'
    fname_ID = infold + ThisExpCond + '_allsubjs_ID_'  + mdltype + '.pickle'
        
    # load pickles here
    with open(fname_X, 'rb') as handle:
        allsubjs_X = pickle.load(handle)
    with open(fname_Y, 'rb') as handle:
        allsubjs_Y = pickle.load(handle)
    with open(fname_ID, 'rb') as handle:
        allsubjs_ID = pickle.load(handle)

    #%% data selection & feature relabeling

    # select the full (aggregate) data for all the analyses that follow    
    X_full = allsubjs_X['aggregate']

    # generate features labels which take into account the recursiveness of 
    # labels due to parcels
    
    feats_unique_labels = list(allsubjs_X.keys())
    # remove "aggregate" 
    feats_unique_labels.remove('aggregate')
    
    # final labeling 
    feats_labels = []

    for ifeat in feats_unique_labels:
        
        this_array = allsubjs_X[ifeat]
        feats_labels.extend([ifeat] * this_array.shape[1])
    
    #%% apply factor analysis
    
    transformer_FA = FactorAnalysis(n_components=2, random_state=42)
    X_transformed_FA = transformer_FA.fit_transform(X_full)
    
    FA_mdls_list.append(transformer_FA)    
    
    
    #%% apply LDA
    
    this_LDA = LinearDiscriminantAnalysis(n_components=1)
    this_LDA.fit(X_full, allsubjs_Y)

    X_transformed_LDA = this_LDA.transform(X_full)
    
    LDA_mdls_list.append(this_LDA)
    
    #%% group & plot loadings according to feature categories -average across parcels
    array_loads = np.empty((2, len(feats_unique_labels)))
    acc_feat = 0
    for ifeat in feats_unique_labels:
    
        mask_ = [s == ifeat for s in feats_labels]
        array_loads[0, acc_feat] = this_LDA.coef_[0, mask_].mean()
        array_loads[1, acc_feat] = transformer_FA.components_[0, mask_].mean()

        acc_feat += 1

    DF_loadings = pd.DataFrame(array_loads.T, columns=['LDA loads', 'FA loads'], 
                               index=feats_unique_labels)
    DF_loadings.plot.barh()
    plt.tight_layout()
    
    
    #%% get single subject transformation on 2d plane
    
    condtypes = [0, 1]; list_SUBJID = list(set(allsubjs_ID))
    
    all_subjs_dict = {'LatentFactor1' : [],
                      'LatentFactor2' : [],
                      'LDA_transformed' : [],
                      'condition' : []}
    
    for SUBJid in list_SUBJID:
        
        # get logical for the current subject    
        lgcl_this_subj = [s == SUBJid for s in allsubjs_ID]
    
        # this_subj_dict = {'comp1' : [],
        #                   'comp2' : [],
        #                   'condition' : []}
    
        for icond in range(2):
            
            lgcl_this_cond = allsubjs_Y == (icond+1)
    
            mask_ = lgcl_this_cond & lgcl_this_subj
            
            this_transition_FA = X_transformed_FA[mask_, :]
            this_transition_LDA = X_transformed_LDA[mask_, 0]
    
            # this_subj_dict['comp1'].extend(list(this_transition[:, 0]))        
            # this_subj_dict['comp2'].extend(list(this_transition[:, 1]))        
            # this_subj_dict['condition'].extend([condnames[icond]] * sum(mask_))        
    
            all_subjs_dict['LatentFactor1'].append(this_transition_FA[:, 0].mean())        
            all_subjs_dict['LatentFactor2'].append(this_transition_FA[:, 1].mean())     
            all_subjs_dict['LDA_transformed'].append(this_transition_LDA.mean())     
            
            all_subjs_dict['condition'].append(condnames[icond])        
    
    
    
        # DF_this_subj = pd.DataFrame.from_dict(this_subj_dict)
        # plt.figure()
        # sns.kdeplot(data=DF_this_subj, x='comp1', y='comp2', hue='condition')
    
    
    DF_all_subjs = pd.DataFrame.from_dict(all_subjs_dict)
    plt.figure()
    plt.subplot(221)
    sns.kdeplot(data=DF_all_subjs, x='LatentFactor1', y='LatentFactor2', hue='condition')
    
    plt.subplot(222)
    plt.scatter(DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[0]], 
                DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[0]], s=20)
    
    plt.scatter(DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[1]], 
                DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[1]], s=20)
    
    plt.plot(np.array([DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[0]], 
             DF_all_subjs['LatentFactor1'].loc[DF_all_subjs['condition']==condnames[1]]]), 
             np.array([DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[0]],
                      DF_all_subjs['LatentFactor2'].loc[DF_all_subjs['condition']==condnames[1]]]),
             'k', linewidth=.5, alpha=.1)
    plt.xlabel('LatentFactor1')
    plt.ylabel('LatentFactor2')
    
    plt.subplot(223)
    sns.kdeplot(data=DF_all_subjs, x='LDA_transformed', hue='condition')
    
    plt.suptitle(ThisExpCond)
    
    acc_expCond +=1
    
    
plt.figure()
plt.subplot(131)
plt.scatter(LDA_mdls_list[0].coef_, LDA_mdls_list[1].coef_, s=10)
plt.title('coeffs LDA')
plt.xlabel('VS')
plt.ylabel('ECEO')

plt.subplot(132)
plt.scatter(FA_mdls_list[0].components_[0, :], 
            FA_mdls_list[1].components_[0, :], s=10)
plt.title('coeffs FA \n(1st factor)')
plt.xlabel('VS')
plt.ylabel('ECEO')

#%% 


    
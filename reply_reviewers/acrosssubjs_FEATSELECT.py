# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
np.random.seed(42)
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os
import copy
# import dask
from datetime import datetime

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score

# pipeline definer
from sklearn.pipeline import Pipeline

# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer

# crossvalidation
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

# imputer
from sklearn.impute import SimpleImputer

# feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# scaler
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer # to be used at some point?


# various 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/rev_reply_MEG/'

# output folder
outfold = '../STRG_decoding_accuracy/rev_reply_MEG/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

expconds = ['ECEO', 'VS'];
mdltypes = ['TimeFeats', 'FullFFT']
n_cvs = 10 

#%%    

cond_list_subj = []

for icond in expconds:

    full_subj_dict = {'X':[], 'Y':[], 'groups':[], 'subj':[], 'single features':{}}
        
    acc_mdl = 0
    for imdl in mdltypes:

        # before looping through participants, fetch one dataset just for the names
        # of the features
        tmpfname = infold + f'{12}_{icond}_{imdl}.mat'
        tmp = loadmat_struct(tmpfname)
        tmp = tmp['variableName']
        featnames = list(tmp['single_feats'].keys())
        
       
        acc_feat = 0 
        for ifeat in featnames:
            
            full_subj_dict['single features'].update({ifeat : []})     
            
            for isubj in range(29):
                
                fname = infold + f'{isubj+1:02d}_{icond}_{imdl}.mat'
    
                mat_content = loadmat_struct(fname)
                F = mat_content['variableName']
                Y_labels = F['Y']
                single_feats = F['single_feats']
                            
                # scaling
                X_scld = RobustScaler().fit_transform(single_feats[ifeat]) 
                    
                # tanh
                X_scld = np.tanh(X_scld)
                
                full_subj_dict['single features'][ifeat].append(X_scld)                
                
                if acc_feat==0 and acc_mdl==0:

                    full_subj_dict['Y'].append(Y_labels[:, np.newaxis])
                    full_subj_dict['subj'] = full_subj_dict['subj'] + [isubj]*len(Y_labels)
              
            full_subj_dict['single features'][ifeat] = np.concatenate(full_subj_dict['single features'][ifeat], axis=0)    

            if acc_feat==0 and acc_mdl==0:
                
                full_subj_dict['Y'] = np.concatenate(full_subj_dict['Y'], 
                                                     axis=0)[:, 0]
            acc_feat+=1
            
            print(f'{icond} {imdl} {ifeat}')
           
        acc_mdl+=1
    
    # concatneate all the features in one big X mat
    acc_feat = 0
    for key, value in full_subj_dict['single features'].items():
        
        if acc_feat==0:
            full_subj_dict['X'] = value
        else:
            full_subj_dict['X'] = np.concatenate((full_subj_dict['X'], value), axis=1)
            
        full_subj_dict['groups']+=[key]*value.shape[1]

        acc_feat+=1

    # feature selection based on SVC
    lsvc = LinearSVC(C=10, penalty="l1", dual=False).fit(full_subj_dict['X'], 
                                                           full_subj_dict['Y']) # C=10 same choice used throughout the paper. C=.01 in tutorials
    model = SelectFromModel(lsvc, prefit=True)
    full_subj_dict.update({'FeatSelector' : copy.deepcopy(model)})
    cond_list_subj.append(full_subj_dict)
    
    # # feature selection based on Random Forest
    # forest = RandomForestClassifier(random_state=42)
    # forest.fit(full_subj_dict['X'], full_subj_dict['Y'])
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    
#%%



plt.figure()
for iplot in range(2):
    
    tmp = cond_list_subj[iplot] 
    TimeFeats_coeffs, FFT_coeffs = copy.deepcopy(tmp['FeatSelector'].estimator.coef_[0, :]), copy.deepcopy(tmp['FeatSelector'].estimator.coef_[0, :])
    TimeFeats_coeffs[np.array(tmp['groups'])=='fullFFT'] = np.nan
    FFT_coeffs[np.array(tmp['groups'])!='fullFFT'] = np.nan
    
    plt.subplot(2, 1, iplot+1)
    
    plt.plot(np.abs(TimeFeats_coeffs))    
    plt.plot(np.abs(FFT_coeffs))
    
    plt.title(expconds[iplot])
    
    
plt.tight_layout()
plt.legend(('TimeFeats', 'fullFFT'))
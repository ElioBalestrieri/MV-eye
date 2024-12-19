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

# pipeline definer
from sklearn.pipeline import Pipeline

# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer

# crossvalidation
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer # to be used at some point?
from sklearn import metrics

# various 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# define the pipeline to be used
# class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
#                                                           strategy='mean')),
#                            ('squeezer', FunctionTransformer(np.tanh)),
#                            ('std_PCA', PCA(n_components=.9, svd_solver='full')),
#                            ('SVM', SVC(C=10))
#                            ]) it used to contain the robustscaler as well, which is now included at the single subj level

SVM_pipeline = Pipeline([('scaler', RobustScaler()),
                         ('squeezer', FunctionTransformer(np.tanh)),
                         ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                         ('SVM', SVC(C=10))
                         ])

LDA_pipeline = Pipeline([('scaler', RobustScaler()),
                         ('squeezer', FunctionTransformer(np.tanh)),
                         ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                         ('LDA', LinearDiscriminantAnalysis())
                         ])

LogReg_pipeline = Pipeline([('scaler', RobustScaler()),
                            ('squeezer', FunctionTransformer(np.tanh)),
                            ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                            ('LogReg', LogisticRegression())
                            ])


#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/rev_reply_MEG_zscore/'

# output folder
outfold = '../STRG_decoding_accuracy/rev_reply_MEG_zscore/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% functions

def MyDecoders(X, Y, groups, ifeat, imdl, icond):
    
    n_cvs = len(np.unique(groups))
    
    # decoding, single feats
    SVM_acc = cross_val_score(SVM_pipeline, X, Y, cv=LeaveOneGroupOut(), 
                              groups=groups,
                              scoring='balanced_accuracy')       


    tmp_acc = list(SVM_acc) 
    tmp_feat = [ifeat]*len(tmp_acc)
    tmp_class = ['SVM']*n_cvs
    tmp_cond = [icond]*len(tmp_acc)
    tmp_mdl = [imdl]*len(tmp_acc)
    tmp_fold = [f'{i}CVpart' for i in range(n_cvs)]
    
    tmp_dict = {'model' : tmp_mdl, 'condition' : tmp_cond, 'feature' : tmp_feat,
                'classifier' : tmp_class, 'accuracy' : tmp_acc, 'cvFolds' : tmp_fold}
    
    tmp_DF = pd.DataFrame.from_dict(tmp_dict)
    
    return tmp_DF


#%%

expconds = ['VS', 'ECEO'];
mdltypes = ['TimeFeats', 'FreqBands', 'FullFFT']
n_cvs = 10 

#%%    

for icond in expconds:
        
    for imdl in mdltypes:

        # before looping through participants, fetch one dataset just for the names
        # of the features
        tmpfname = infold + f'{12}_{icond}_{imdl}.mat'
        tmp = loadmat_struct(tmpfname)
        tmp = tmp['variableName']
        featnames = list(tmp['single_feats'].keys())
        
        full_subj_dict = {'X':[], 'Y':[], 'groups':[], 'single features' : {}}
        
        acc_feat = 0 
        for ifeat in featnames:
            
            full_subj_dict['single features'].update({ifeat : []})     
            
            if (ifeat == 'mean') or (ifeat == 'std'):
                
                continue
            
            
            for isubj in range(29):
                
                fname = infold + f'{isubj+1:02d}_{icond}_{imdl}.mat'
    
                mat_content = loadmat_struct(fname)
                F = mat_content['variableName']
                Y_labels = F['Y']
                single_feats = F['single_feats']
            
                scaled_X = single_feats[ifeat] # misnomer, since the scaling happens later. only her efor quick edit
                
                full_subj_dict['single features'][ifeat].append(scaled_X)
                
                if acc_feat==0:

                    groups = KBinsDiscretizer(n_bins=n_cvs, 
                                              encode='ordinal', 
                                              strategy='uniform').fit_transform(F['trl_order'][:,np.newaxis])[:, 0]
                    full_subj_dict['Y'].append(Y_labels[:, np.newaxis])
                    full_subj_dict['groups'].append(groups[:, np.newaxis])
                
            full_subj_dict['single features'][ifeat] = np.concatenate(full_subj_dict['single features'][ifeat], axis=0)    

            if acc_feat==0:
                
                full_subj_dict['Y'] = np.concatenate(full_subj_dict['Y'], 
                                                     axis=0)[:, 0]
                full_subj_dict['groups'] = np.concatenate(full_subj_dict['groups'], 
                                                          axis=0)[:, 0]
            
            tmp_DF = MyDecoders(full_subj_dict['single features'][ifeat], 
                                full_subj_dict['Y'], full_subj_dict['groups'],
                                ifeat, imdl, icond)                        
            
            if (acc_feat==0) & (imdl==mdltypes[0]) & (icond==expconds[0]): # concatenate all accuracy data in a big dataframe
                
                stable_DF = copy.deepcopy(tmp_DF)
                
            else:
                
                stable_DF = pd.concat((stable_DF, tmp_DF))                

            stable_DF.to_csv(outfold + 'across_accuracy_TMP_ZSCORED.csv', index=False)
            full_subj_dict['X'].append([full_subj_dict['single features'][ifeat]])

            acc_feat+=1
            
            print(f'{icond} {imdl} {ifeat}')
            
        # full set
        if imdl!='FullFFT':
            
            bigX = np.concatenate(full_subj_dict['X'], axis=2)[0, :, :]
            
            tmp_DF = MyDecoders(bigX, 
                                full_subj_dict['Y'], full_subj_dict['groups'],
                                'full set', imdl, icond)                        
            
            stable_DF = pd.concat((stable_DF, tmp_DF))          
            print(f'{icond} {imdl} full set')
        
#%%    

stable_DF.to_csv(outfold + 'across_accuracy_FINAL_ZSCORED.csv', index=False)
            
        
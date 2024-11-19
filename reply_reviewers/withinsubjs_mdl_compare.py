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
# import dask
from datetime import datetime
import copy

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

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
from sklearn.metrics import balanced_accuracy_score

# various 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# define the pipeline to be used
class_pipeline = Pipeline([
                           ('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10))
                           ])

# ('inpute_missing', SimpleImputer(missing_values=np.nan, strategy='mean')),

#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/rev_reply_MEG/'

# output folder
outfold = '../STRG_decoding_accuracy/rev_reply_MEG/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% functions

def loop_classify_feats(Xs, Y, pipe, groups, debugmode=False):

    storing_mat = {}
    
    count_exceptions = 0
    acc_feat = 0; 

    for key, X in Xs.items():
                
        # get rid of full nan columns, if any
        X = X[:, ~(np.any(np.isnan(X), axis=0))]
    
        # # start concatenate arrays for whole freqband decode #################### old approach
        if acc_feat==0:
            catX = np.copy(X)
        else:
            catX = np.concatenate((catX, X), axis=1)
        
        # try decoding (supervised learning)
        try:        
            acc = cross_val_score(pipe, X, Y, cv=LeaveOneGroupOut(), 
                                  groups=groups,
                                  scoring='balanced_accuracy').mean()        
            if debugmode:
                print(key)
            
        except:
            acc = np.nan; count_exceptions += 1
            
        # append feat-
        storing_mat.update({key : [acc]}) 
        
        acc_feat += 1    
     
    # compute accuracy on the aggregated features.
    if not key=='FullFFT':

        try:

            acc_whole = cross_val_score(pipe, catX, Y, cv=LeaveOneGroupOut(), 
                                        groups=groups,
                                        scoring='balanced_accuracy').mean()
            
        except:
            
            acc_whole = np.nan; count_exceptions += 1

        storing_mat.update({'full_set' : [acc_whole]}) 

  
    subjDF = pd.DataFrame.from_dict(storing_mat)
    
    return subjDF, count_exceptions
    
#%%

expconds = ['ECEO', 'VS'];
mdltypes = ['TimeFeats']# , 'FreqBands', 'FullFFT']
n_cvs = 10 

for isubj in range(30):
    
    for icond in expconds:
        
        for imdl in mdltypes:
            
            fname = infold + f'{isubj+1:02d}_{icond}_{imdl}.mat'

            mat_content = loadmat_struct(fname)
            F = mat_content['variableName']
            Y_labels = F['Y']
            single_feats = F['single_feats']
            
            # load fooof comparison data
            fname_2 = infold + f'aperiodic_only/{isubj+1:02d}_{icond}_{imdl}_FOOOFcompare.mat'

            mat_content2 = loadmat_struct(fname_2)
            F2 = mat_content2['variableName']
            single_feats2 = F2['single_feats']
            
            del single_feats['fooof_slope'], single_feats['fooof_offset']
            
            single_feats['custom_slope'] = single_feats2['custom_slope']
            single_feats['custom_offset'] = single_feats2['custom_offset']

            groups = KBinsDiscretizer(n_bins=n_cvs, 
                                      encode='ordinal', 
                                      strategy='uniform').fit_transform(F['trl_order'][:,np.newaxis])[:, 0]

            subjDF, count_excs = loop_classify_feats(single_feats, Y_labels, 
                                                     class_pipeline, groups)

            subjDF.to_csv(outfold + f'{isubj+1:02d}_{icond}_{imdl}_customFOOOF.csv', index=False)
            
            print(f'{isubj+1:02d} {icond} {imdl}', end='\r')
            
#%% old stuff

# # 
# for igroup in np.unique(groups):

#     mask_group = groups == igroup
#     tmp_Xtr = X[mask_group, :]
#     tmp_Xte = X[~mask_group, :]
#     tmp_Ytr = Y[mask_group]
#     tmp_Yte = Y[~mask_group]

#     # scaling
#     sclr = RobustScaler().fit(tmp_Xtr) 
#     tmp_Xtr = sclr.transform(tmp_Xtr)
#     tmp_Xte = sclr.transform(tmp_Xte)
        
#     # tanh
#     tmp_Xtr = np.tanh(tmp_Xtr)
#     tmp_Xte = np.tanh(tmp_Xte)

#     # PCA
#     PCA_mdl = PCA(n_components=.9, svd_solver='full').fit(tmp_Xtr) 
#     tmp_Xtr = PCA_mdl.transform(tmp_Xtr)
#     tmp_Xte = PCA_mdl.transform(tmp_Xte)

#     if acc_feat==0:
    
#         tmp_dict = {'X_train' : [], 'X_test' : [], 
#                 'Y_train' : tmp_Ytr, 'Y_test' : tmp_Yte}
#         list_splits.append(copy.deepcopy(tmp_dict))


#     idx = int(igroup)
#     list_splits[idx]['X_train'].append(tmp_Xtr)
#     list_splits[idx]['X_test'].append(tmp_Xte)






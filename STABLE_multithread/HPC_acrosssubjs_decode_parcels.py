# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general tools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os
import dask

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test

#%% folder(s) definition & cond type

#input folder 
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# print(sys.argv)
ThisExpCond = sys.argv[1]

# load parcellation labels, and select only visual cortices
HCP_parcels = pd.read_csv('../helper_functions/HCP-MMP1_UniqueRegionList_RL.csv')
red_HCP = HCP_parcels[HCP_parcels['cortex'].str.contains('visual', case=False)]

#%% classification tools
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

#%% models and conditions
# ExpConds = ['ECEO', 'VS']
mdltypes = ['FullFFT', 'TimeFeats', 'FTM', 'FreqBands']

#%% define function to be run in parallel with dask
@dask.delayed
def decode_parcel(X_train, y_train, X_test, y_test):
    
    mdl_ = Pipeline([('std_PCA', PCA(n_components=.9, svd_solver='full')),
                      ('SVM', SVC(C=10))])
                  
    mdl_ = mdl_.fit(X_train, y_train)
    predictions = mdl_.predict(X_test)
    prcl_accuracy = balanced_accuracy_score(y_test, predictions)
        
    return prcl_accuracy

#%% inititate loop
n_parcels = 52
for imdl in mdltypes:

    data_type = ThisExpCond + '_' + imdl
    fullX_train, fullX_test, Y_train, Y_test = cat_subjs_train_test(infold, strtsubj=0, endsubj=29, 
                                                            ftype=data_type, tanh_flag=True, 
                                                            compress_flag=True)
    ntrls_test = len(Y_test)
    ntrls_train = len(Y_train)
    n_feats = int(fullX_train['aggregate'].shape[1]/n_parcels)
    
    parc_X_train = np.reshape(fullX_train['aggregate'], 
                              (ntrls_train, n_parcels, n_feats), order='F')
    parc_X_test = np.reshape(fullX_test['aggregate'], 
                             (ntrls_test, n_parcels, n_feats), order='F')
    # parallel loop
    parcels_accs = []
    for iprcl in range(n_parcels):
        
        this_Xtrain = parc_X_train[:, iprcl, :]
        this_Xtest = parc_X_test[:, iprcl, :]
        
        parcels_accs.append(decode_parcel(this_Xtrain, Y_train, 
                                          this_Xtest, Y_test))
        
    # execute parallel loop
    accs_tuple = dask.compute(parcels_accs)
    accs_this_cond = accs_tuple[0]
    # append output to parcels table
    red_HCP[imdl + '_decoding_accuracy'] = accs_this_cond

# save output
foutname_parcels = outfold + 'AcrossSubjs_' + ThisExpCond + '_' + '_SingleParcels_accuracy.csv'
red_HCP.to_csv(foutname_parcels)
    
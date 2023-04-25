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


#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test


#%% folder(s) definition & cond type

#input folder 
infold = '../STRG_computed_features/Mdl_comparison/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# print(sys.argv)
# ThisExpCond = sys.argv[1]


#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# pipeline definer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

# define pipeline object
pipe = Pipeline([
                 ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                 ('SVM', SVC(C=10))
                ])

from sklearn.feature_selection import SequentialFeatureSelector

rbf_svm = SVC(C=10, verbose=True)


#%% models and conditions
ExpConds = ['ECEO', 'VS']
mdltypes = ['FullFFT', 'TimeFeats', 'FTM', 'FreqBands']

#%% non parallel part

# concatenate files between participant, after within-participant normalization

acc_type = 0
full_count_exc = 0

for ThisExpCond in ExpConds:
        
    for imdl in mdltypes:
    
        data_type = ThisExpCond + '_' + imdl
        fullX_train, fullX_test, Y_train, Y_test = cat_subjs_train_test(infold, strtsubj=0, endsubj=29, 
                                                                ftype=data_type, tanh_flag=True, 
                                                                compress_flag=True)
        
        # preallocate matrix (if first iteration on the loop)
        mat_accs = np.empty((1, len(fullX_train)))
            
        acc_feat = 0
        for key in fullX_train:
            
            X_train = fullX_train[key]
            X_test = fullX_test[key]
                        
            mdl_ = pipe.fit(X_train, y=Y_train)
                    
            mdl_prediction = mdl_.predict(X_test)
                    
            LeftOut_acc = balanced_accuracy_score(Y_test, mdl_prediction)
                       
            mat_accs[0, acc_feat] = LeftOut_acc
                          
            acc_feat+=1
            
            print(key)
                      
        DF = pd.DataFrame(data=mat_accs, columns=fullX_train.keys(), index=[imdl])
                  
        # save
        fname_out = outfold + 'AcrossSubjs_' + ThisExpCond +  '_' + imdl + '.csv'
        DF.to_csv(fname_out)
        
        

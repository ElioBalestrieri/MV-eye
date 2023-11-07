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
import os, fnmatch
import re

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test


#%% folder(s) definition & cond type

#input folder 
infold = '../STRG_computed_features/LEMON/'

# output folder
outfold = '../STRG_decoding_accuracy/LEMON/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# print(sys.argv)
# ThisExpCond = sys.argv[1]


#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# pipeline definer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

# define pipeline object
# pipe = Pipeline([
#                  ('std_PCA', PCA(n_components=.9, svd_solver='full')),
#                  ('SVM', SVC(C=10))
#                 ])

pipe = Pipeline([('scaler', StandardScaler()),
                 ('LinearSVM', LinearSVC(dual=False))
                ])

#%% models and conditions
mdltypes = ['FullFFT', 'TimeFeats', 'FTM', 'FreqBands']

#%% non parallel part

# concatenate files between participant, after within-participant normalization
acc_type = 0
full_count_exc = 0

# compile regex for further PC selection in keys
r = re.compile("PC_full_set") 
        
for imdl in mdltypes:
    
    data_type = imdl
    filt_fnames = fnmatch.filter(os.listdir(infold), 'sub-*' + imdl + '*.mat')
    acc = 0
    for iname in filt_fnames: 
        iname = iname[0:10]
        filt_fnames[acc] = iname
        acc +=1
    
    fullX_train, fullX_test, Y_train, Y_test, subjID_trials_labels = cat_subjs_train_test(infold, subjlist=filt_fnames, 
                                                            ftype=data_type, tanh_flag=True, 
                                                            compress_flag=True,
                                                            pca_kept_var=.9)
    
    PC_list_names = newlist = list(filter(r.match, fullX_train.keys()))
        
    # preallocate matrix (if first iteration on the loop)
    mat_accs = np.empty((1, len(PC_list_names)))
        
    all_PC_names = 0 
    
    
    acc_feat = 0
    for key in PC_list_names:
        
        X_train = fullX_train[key]
        X_test = fullX_test[key]
                    
        mdl_ = pipe.fit(X_train, y=Y_train)
                
        mdl_prediction = mdl_.predict(X_test)
                
        LeftOut_acc = balanced_accuracy_score(Y_test, mdl_prediction)
                   
        mat_accs[0, acc_feat] = LeftOut_acc
                      
        acc_feat+=1
        
        print(key)
                  
    DF = pd.DataFrame(data=mat_accs, columns=PC_list_names, index=[imdl])
              
    # save
    fname_out = outfold + 'AcrossSubjs_PC_LinearSVM_' + imdl + '.csv'
    DF.to_csv(fname_out)
        
        

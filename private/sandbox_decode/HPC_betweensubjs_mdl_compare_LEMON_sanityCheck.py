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
import dask
from datetime import datetime
import pickle


#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import split_subjs_train_test

#%% folder(s) definition & cond type

#input folder 
infold = '../STRG_computed_features/LEMON/'

# output folder
outfold = '../STRG_decoding_accuracy/LEMON/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

#%% Classification object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# pipeline definer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

#%% create partitions
n_folds = 5

#%% non parallel part

# concatenate files between participant, after within-participant normalization

mdltypes = ['FreqBands', 'TimeFeats', 'FullFFT', 'FTM']
acc_type = 0
full_count_exc = 0
    
CPU_jobs = []    
for imdl in mdltypes:

    for ifold in range(n_folds):
    
        jobcode = 'Fold_' + str(ifold) + '_' + imdl
        
        fname_X_train = infold + jobcode + '_X_train.pickle'
        fname_Y_train = infold + jobcode + '_Y_train.pickle'
        fname_X_test = infold + jobcode + '_X_test.pickle'
        fname_Y_test = infold + jobcode + '_Y_test.pickle'
        
        # load pickles here
        with open(fname_X_train, 'rb') as handle:
            X_train = pickle.load(handle)
        # load pickles here
        with open(fname_Y_train, 'rb') as handle:
            Y_train = pickle.load(handle)
        # load pickles here
        with open(fname_X_test, 'rb') as handle:
            X_test = pickle.load(handle)
        # load pickles here
        with open(fname_Y_test, 'rb') as handle:
            Y_test = pickle.load(handle)
                
        print('\n\n\n\n#######    ' + imdl + '     #######\n')        
        feed_ = 'X_train' + str(X_train.shape) + '\nY_train' + str(Y_train.shape) + '\nX_test' + str(X_test.shape) + '\nY_test' + str(Y_test.shape)
        print(feed_)
        
        

# #%% parallel

# # define the parallel function to be called in separate threads
# @dask.delayed
# def decode_onefold_onemodel(jobcode, infold, outfold):

#     pipe = Pipeline([
#                      ('SVM', SVC(C=10))
#                     ])

#     fname_X_train = infold + jobcode + '_X_train.pickle'
#     fname_Y_train = infold + jobcode + '_Y_train.pickle'
#     fname_X_test = infold + jobcode + '_X_test.pickle'
#     fname_Y_test = infold + jobcode + '_Y_test.pickle'
    
#     # load pickles here
#     with open(fname_X_train, 'rb') as handle:
#         X_train = pickle.load(handle)
#     # load pickles here
#     with open(fname_Y_train, 'rb') as handle:
#         Y_train = pickle.load(handle)
#     # load pickles here
#     with open(fname_X_test, 'rb') as handle:
#         X_test = pickle.load(handle)
#     # load pickles here
#     with open(fname_Y_test, 'rb') as handle:
#         Y_test = pickle.load(handle)
            
#     mdl_ = pipe.fit(X_train, y=Y_train)
            
#     predict_LO = mdl_.predict(X_test)            
#     LO_acc = balanced_accuracy_score(Y_test, predict_LO)
               
#     DF = pd.DataFrame(data=LO_acc, columns=['test_accuracy'], index=[jobcode])
          
#     # save
#     fname_out = outfold + jobcode + '_BS_fullset.csv'
#     DF.to_csv(fname_out)
        
#     return {jobcode : LO_acc}
          

# #%% loop pipeline across subjects and features

# allFolds_DFs = []
# for jobcode in CPU_jobs:

#     joblisted = decode_onefold_onemodel(jobcode, infold, outfold)
#     allFolds_DFs.append(joblisted)
    
# #%% actually launch process
# out = dask.compute(allFolds_DFs)

# #%% log and save 
# list_out = list(out)
# dateTimeObj = datetime.now()
# fname = 'log_' + str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_' + str(dateTimeObj.hour) + '.pickle'

# with open(fname, 'wb') as handle:
#     pickle.dump(list_out, handle, protocol=pickle.HIGHEST_PROTOCOL)



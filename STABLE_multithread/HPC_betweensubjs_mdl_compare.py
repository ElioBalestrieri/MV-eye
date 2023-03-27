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
from mv_python_utils import cat_subjs



#%% folder(s) definition & cond type

#input folder 
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

print(sys.argv)
ThisExpCond = sys.argv[1]

#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# pipeline definer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

#%% non parallel part

# concatenate files between participant, after within-participant normalization

mdltypes = ['FullFFT'] # 'FreqBands', 'FullFFT', 'TimeFeats'
acc_type = 0
full_count_exc = 0
        
for imdl in mdltypes:
    
    data_type = ThisExpCond + '_' + imdl
    allsubjs_X, allsubjs_Y, allsubjs_ID = cat_subjs(infold, strtsubj=0, endsubj=29, 
                                                    ftype=data_type, tanh_flag=True)
    
    fname_X = infold + ThisExpCond + '_allsubjs_X_'  + imdl + '.pickle'
    fname_Y = infold + ThisExpCond + '_allsubjs_Y_'  + imdl + '.pickle'
    fname_ID = infold + ThisExpCond + '_allsubjs_ID_'  + imdl + '.pickle'

    with open(fname_X, 'wb') as handle:
        pickle.dump(np.float32(allsubjs_X), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fname_Y, 'wb') as handle:
        pickle.dump(allsubjs_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fname_ID, 'wb') as handle:
        pickle.dump(allsubjs_ID, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% parallel

# define the parallel function to be called in separate threads
@dask.delayed
def decode_leave_one_subj_out(isubj, infold, ThisExpCond, mdltypes):

    pipe = Pipeline([
                     ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                     ('SVM', SVC(C=10))
                    ])

    SUBJid = f'ID_{isubj+1:02d}' 
    

    acc_mdl = 0
    for imdl in mdltypes:
                
        fname_X = infold + ThisExpCond + '_allsubjs_X_'  + imdl + '.pickle'
        fname_Y = infold + ThisExpCond + '_allsubjs_Y_'  + imdl + '.pickle'
        fname_ID = infold + ThisExpCond + '_allsubjs_ID_'  + imdl + '.pickle'
    
        # load pickles here
        with open(fname_X, 'rb') as handle:
            allsubjs_X = pickle.load(handle)
        with open(fname_Y, 'rb') as handle:
            allsubjs_Y = pickle.load(handle)
        with open(fname_ID, 'rb') as handle:
            allsubjs_ID = pickle.load(handle)

        # preallocate matrix (if first iteration on the loop)
        mat_accs = np.empty((1, len(allsubjs_X)))

        # leave the current subject out for testing    
        lgcl_test = [s == SUBJid for s in allsubjs_ID]
        lgcl_train = [s != SUBJid for s in allsubjs_ID]
        
        Y_train = allsubjs_Y[lgcl_train]
        Y_test = allsubjs_Y[lgcl_test]
    
        acc_feat = 0
        for key in allsubjs_X:
    
            X_ = allsubjs_X[key]
    
            X_train = X_[lgcl_train, :]
            X_test = X_[lgcl_test, :]
            
            mdl_ = pipe.fit(X_train, y=Y_train)
            
            predict_LOS = mdl_.predict(X_test)
            
            LOS_acc = balanced_accuracy_score(Y_test, predict_LOS)
               
            mat_accs[0, acc_feat] = LOS_acc
                  
            acc_feat+=1
              
        DF = pd.DataFrame(data=mat_accs, columns=allsubjs_X.keys(), index=[imdl])
          
        # save
        fname_out = '../STRG_decoding_accuracy/' + SUBJid + '_leftout_' + ThisExpCond +  '_' + imdl + '_intersubjs_accs.csv'
        DF.to_csv(fname_out)
        
    return SUBJid
          

#%% loop pipeline across subjects and features

nsubjs_loaded = 29

allsubjs_DFs = []
for isubj in range(nsubjs_loaded):

    subjID = decode_leave_one_subj_out(isubj, infold, ThisExpCond, mdltypes)
    allsubjs_DFs.append(subjID)
    
#%% actually launch process
countExc = dask.compute(allsubjs_DFs)

#%% log and save 
countExcDF = pd.DataFrame(list(countExc))
dateTimeObj = datetime.now()
fname = 'log_' + str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_' + str(dateTimeObj.hour) + '.csv'
countExcDF.to_csv(fname)

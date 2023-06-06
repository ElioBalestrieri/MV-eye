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
from mv_python_utils import cat_subjs


#%% folder(s) definition & cond type

#input folder 
infold = '../STRG_computed_features/LEMON/'

# output folder
outfold = '../STRG_decoding_accuracy/LEMON/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# pipeline definer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

#%% non parallel part

# concatenate files between participant, after within-participant normalization

mdltypes = ['FreqBands', 'TimeFeats'] # , 'FullFFT', 'FTM']
acc_type = 0
full_count_exc = 0
        
for imdl in mdltypes:
    
    filt_fnames = fnmatch.filter(os.listdir(infold), '*' + imdl +'*')
    acc = 0
    for iname in filt_fnames: 
        iname = iname[0:10]
        filt_fnames[acc] = iname
        acc +=1
    
    allsubjs_X, allsubjs_Y, allsubjs_ID, full_trl_order = cat_subjs(infold, subjlist=filt_fnames,
                                                    ftype=imdl, tanh_flag=True, 
                                                    compress_flag=True,
                                                    all_feats_flag=False) # newly adde dto just concatenate full_set
    
    fname_X = infold + 'Allsubjs_X_'  + imdl + '_fullset.pickle'
    fname_Y = infold + 'Allsubjs_Y_'  + imdl + '.pickle'
    fname_ID = infold + 'Allsubjs_ID_'  + imdl + '.pickle'

    with open(fname_X, 'wb') as handle:
        pickle.dump(allsubjs_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fname_Y, 'wb') as handle:
        pickle.dump(allsubjs_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(fname_ID, 'wb') as handle:
        pickle.dump(allsubjs_ID, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% parallel

# define the parallel function to be called in separate threads
@dask.delayed
def decode_leave_one_subj_out(isubj, infold, mdltypes, outfold):

    pipe = Pipeline([
                     ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                     ('SVM', SVC(C=10))
                    ])

    SUBJid = isubj
    
    for imdl in mdltypes:
                
        fname_X = infold + 'Allsubjs_X_'  + imdl + '_fullset.pickle'
        fname_Y = infold + 'Allsubjs_Y_'  + imdl + '.pickle'
        fname_ID = infold + 'Allsubjs_ID_'  + imdl + '.pickle'
    
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
        fname_out = outfold + SUBJid + '_leftout_' + imdl + '_BS_fullset.csv'
        DF.to_csv(fname_out)
        
    return SUBJid
          

#%% loop pipeline across subjects and features

tmp_mdl = 'FTM' # all subj names are the same in each model condition

allsubjs_DFs = []
for isubj in filt_fnames:

    subjID = decode_leave_one_subj_out(isubj, infold, mdltypes, outfold)
    allsubjs_DFs.append(subjID)
    
#%% actually launch process
countExc = dask.compute(allsubjs_DFs)

#%% log and save 
countExcDF = pd.DataFrame(list(countExc))
dateTimeObj = datetime.now()
fname = 'log_' + str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_' + str(dateTimeObj.hour) + '.csv'
countExcDF.to_csv(fname)
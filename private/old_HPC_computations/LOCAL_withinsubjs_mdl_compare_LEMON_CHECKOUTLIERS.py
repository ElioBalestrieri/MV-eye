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
from datetime import datetime


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
from sklearn.model_selection import cross_val_score

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler # to be used at some point?

# define the pipeline to be used
class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
                                                            strategy='mean')),
                           ('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10))
                           ])


#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/LEMON/'

# output folder
outfold = '../STRG_decoding_accuracy/LEMON/outliers/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% functions

def loop_classify_feats(Xs, Y, pipe, cv_fold=10):
    
    depvars_types = ['decode_acc', 'clust_acc', 'clusts_N']
    storing_mat = np.empty((len(depvars_types), len(Xs)+1))

    count_exceptions = 0
    acc_feat = 0; 
    for key in Xs:
        
        X = Xs[key]
        
        # get rid of full nan columns, if any
        X = X[:, ~(np.any(np.isnan(X), axis=0))]

        # start concatenate arrays for whole feats decode
        if acc_feat==0:
            catX = np.copy(X)
        else:
            catX = np.concatenate((catX, X), axis=1)
        
        acc = np.nan
            
        # UPDATE MATRIX
        storing_mat[0, acc_feat] = acc 
                         
        # backward compatibility to attempted parcellation of accuracy. Add nans as fillers
        acc_clusts = np.nan; 
        N_comps = np.nan

        # UPDATE MATRIX
        storing_mat[1, acc_feat] = acc_clusts
        storing_mat[2, acc_feat] = N_comps
        
        
        acc_feat += 1    
 
    # compute accuracy on the whole freqbands, aggregated features. Always with
    # a try statement for the same reason listed above
    try:
        acc_whole = cross_val_score(pipe, catX, Y, cv=cv_fold, 
                                    scoring='balanced_accuracy').mean()
    except:
        acc_whole = np.nan; count_exceptions += 1

    # feat index updated on last iter
    storing_mat[0, acc_feat] = acc_whole#%% filter out subject for which there was a convergence fail.

    acc_clusts_whole = np.nan; 
    N_comps_whole = np.nan

    storing_mat[1, acc_feat] = acc_clusts_whole
    storing_mat[2, acc_feat] = N_comps_whole

    updtd_col_list = list(Xs.keys()); updtd_col_list.append('full_set')
    
    subjDF_feats = pd.DataFrame(storing_mat, columns=updtd_col_list, index=depvars_types)

    return subjDF_feats, count_exceptions


def single_subj_classify(isubj, infold, outfold):

    
    acc_type = 0
    full_count_exc = 0        
    
    mdltypes = ['TimeFeats']
    
    for imdl in mdltypes:

        fname = infold + isubj + '_' + imdl + '.mat'
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
        Y_labels = F['Y']
        single_feats = F['single_feats']
    
        # call loop across all features + aggregated feature
        subjDF_feats, count_exc1 = loop_classify_feats(single_feats, Y_labels,
                                                         class_pipeline)
        acc_type += 1
    
        foutname_feats = outfold + isubj + '_' + imdl + '_WS.csv'
    
        subjDF_feats.to_csv(foutname_feats)
    
        full_count_exc = full_count_exc + count_exc1

    return full_count_exc

#%% select subjects where the decoding was not successfull
infold_computed = '../STRG_decoding_accuracy/LEMON/'

MdlTypes = ['TimeFeats']# , 'FTM'] #, 'FB_cherrypicked', 'cherrypicked']

filt_fnames = fnmatch.filter(os.listdir(infold_computed), 'sub-*' + MdlTypes[0] +'*')
acc = 0
for iname in filt_fnames: 
    iname = iname[0:10]
    filt_fnames[acc] = iname
    acc +=1


allsubjs_within_DFs = []

for isubj in filt_fnames:
        
    for iMdl in MdlTypes:
    
        subjname = isubj
                    
        # within subjects            
        fname_within = infold_computed + subjname + '_' + iMdl + '_WS.csv'
        DF_WN = pd.read_csv(fname_within)
        
        tmp = DF_WN.set_index('Unnamed: 0').stack().to_frame().\
                    reset_index().rename(columns={'Unnamed: 0': 'freq_band', 
                                                  'level_1': 'feature', 
                                                  0: 'decoding accuracy'})         
        tmp = tmp.drop('freq_band', axis=1)
        
        # add columns
        tmp['subjID'] = [subjname] * tmp.shape[0]
        tmp['model'] = [iMdl] * tmp.shape[0]

        allsubjs_within_DFs.append(tmp)

             
DF_fullsample_WN = pd.concat(allsubjs_within_DFs, ignore_index=True) 

summary_subjcount = DF_fullsample_WN.groupby('subjID').count()
bad_apples = summary_subjcount['feature']<41
bad_apples = list(bad_apples[bad_apples].index)

#%% launch classifiers on bad apples

allsubjs_DFs = []
for isubj in bad_apples:

    outDF = single_subj_classify(isubj, infold, outfold)
    allsubjs_DFs.append(outDF)
    

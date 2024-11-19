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
mdltypes = ['TimeFeats', 'FreqBands', 'FullFFT']
classifiers = ['SVM', 'LDA', 'LogReg']
n_cvs = 10 

#%%    

full_dict = {'model' : [], 'condition' : [], 'feature' : [],
             'classifier' : [], 'accuracy' : [], 'cvFolds' : []}                

for icond in expconds:
        
    for imdl in mdltypes:

        # before looping through participants, fetch one dataset just for the names
        # of the features
        tmpfname = infold + f'{12}_{icond}_{imdl}.mat'
        tmp = loadmat_struct(tmpfname)
        tmp = tmp['variableName']
        featnames = list(tmp['single_feats'].keys())
        
        if imdl =='TimeFeats':
        
            featnames[np.where(np.array(featnames)=='fooof_offset')[0][0]] = 'custom_offset'
            featnames[np.where(np.array(featnames)=='fooof_slope')[0][0]] = 'custom_slope'
                
        full_subj_dict = {'X':[], 'Y':[], 'groups':[], 'subj':[], 'single features':{}}
       
        acc_feat = 0 
        for ifeat in featnames:
            
            full_subj_dict['single features'].update({ifeat : []})     
            
            for isubj in range(29):
                
                fname = infold + f'{isubj+1:02d}_{icond}_{imdl}.mat'
    
                mat_content = loadmat_struct(fname)
                F = mat_content['variableName']
                Y_labels = F['Y']
                single_feats = F['single_feats']
                
                if imdl == 'TimeFeats' and (ifeat=='custom_slope' or ifeat=='custom_offset'):
                    
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
                            
                full_subj_dict['single features'][ifeat].append(single_feats[ifeat])                
                
                if acc_feat==0:

                    full_subj_dict['Y'].append(Y_labels[:, np.newaxis])
                    full_subj_dict['groups'].append(groups[:, np.newaxis])
                    full_subj_dict['subj'] = full_subj_dict['subj'] + [isubj]*len(Y_labels)
              
            full_subj_dict['single features'][ifeat] = np.concatenate(full_subj_dict['single features'][ifeat], axis=0)    

            if acc_feat==0:
                
                full_subj_dict['Y'] = np.concatenate(full_subj_dict['Y'], 
                                                     axis=0)[:, 0]
                full_subj_dict['groups'] = np.concatenate(full_subj_dict['groups'], 
                                                          axis=0)[:, 0]
            
            acc_feat+=1
            
            print(f'{icond} {imdl} {ifeat}')
            
        # from here on, compute recursive scaling of each single feature, CAT and clssify the full model for n folds
        foo = 1
        
        for ifold in range(n_cvs):
            
            tmp_dict = {'X_train' : [], 'X_test' : [], 
                        'Y_train' : [], 'Y_test' : []}
            
            subj_mask = np.array(full_subj_dict['subj'])

            all_subjs_dict = {'X_tr' : {}, 'X_te' : {}}
            for isubj in range(29):

                fold_mask = np.array(full_subj_dict['groups'])[subj_mask==isubj]    
                tmp_Y = full_subj_dict['Y'][subj_mask==isubj]
                
                tmp_dict['Y_train']+=list(tmp_Y[fold_mask!=ifold])
                tmp_dict['Y_test']+=list(tmp_Y[fold_mask==ifold])

                for ifeat in featnames:
                    
                    tmp_X = full_subj_dict['single features'][ifeat][subj_mask==isubj, :]
            
                    X_te = tmp_X[fold_mask==ifold, :]
                    X_tr = tmp_X[fold_mask!=ifold, :]

                    # !IMPORTANT! PREPROCESSING PIPELINE       
                    
                    # scaling
                    sclr = RobustScaler().fit(X_tr) 
                    X_tr = sclr.transform(X_tr)
                    X_te = sclr.transform(X_te)
                    
                    # tanh
                    X_tr = np.tanh(X_tr)
                    X_te = np.tanh(X_te)

                    if isubj == 0:
                        
                        all_subjs_dict['X_tr'].update({ifeat : X_tr})
                        all_subjs_dict['X_te'].update({ifeat : X_te})

                    else:
                        
                        all_subjs_dict['X_tr'][ifeat] = np.concatenate((all_subjs_dict['X_tr'][ifeat], X_tr), axis=0)
                        all_subjs_dict['X_te'][ifeat] = np.concatenate((all_subjs_dict['X_te'][ifeat], X_te), axis=0)
            
            acc_feat = 0                        
            for ifeat in featnames:
                
                X_tr = all_subjs_dict['X_tr'][ifeat]
                X_te = all_subjs_dict['X_te'][ifeat]

                pca_mdl = PCA(n_components=.9, svd_solver='full').fit(X_tr)
                X_tr = pca_mdl.transform(X_tr)
                X_te = pca_mdl.transform(X_te)                    
                
                if acc_feat == 0:
                    
                    tmp_dict['X_train'] = X_tr
                    tmp_dict['X_test'] = X_te

                else:
                    
                    tmp_dict['X_train'] = np.concatenate((tmp_dict['X_train'], X_tr), axis=1)
                    tmp_dict['X_test'] = np.concatenate((tmp_dict['X_test'], X_te), axis=1)
                    
                acc_feat += 1
            
            # PCA on cat feats
            # ... apparently performing PCA at single feats or concat feats 
            # does not change a big deal
            # pca_mdl = PCA(n_components=.9, svd_solver='full').fit(tmp_dict['X_train'])
            # tmp_dict['X_train'] = pca_mdl.transform(tmp_dict['X_train'])
            # tmp_dict['X_test'] = pca_mdl.transform(tmp_dict['X_test'])                                
            
            for iclass in classifiers:
                
                match iclass:                    
                    case 'SVM':
                        class_mdl = SVC(C=10)
                    case 'LDA':
                        class_mdl = LinearDiscriminantAnalysis()
                    case 'LogReg':
                        class_mdl = LogisticRegression()
                        
                class_mdl.fit(tmp_dict['X_train'], tmp_dict['Y_train'])
                preds_ = class_mdl.predict(tmp_dict['X_test'])
                
                acc_class = balanced_accuracy_score(tmp_dict['Y_test'], preds_)            
                # prec_class = precision_score(tmp_dict['Y_test'], preds_)
                # rec_class = recall_score(tmp_dict['Y_test'], preds_)
                # auc_class = roc_auc_score(tmp_dict['Y_test'], preds_)

                
                full_dict['model'].append(imdl)
                full_dict['condition'].append(icond)
                full_dict['classifier'].append(iclass)
                
                full_dict['accuracy'].append(acc_class)                
                # full_dict['precision'].append(prec_class)
                # full_dict['recall'].append(rec_class)
                # full_dict['auc'].append(auc_class)
                
                full_dict['cvFolds'].append(ifold)
                full_dict['feature'].append('full_set')

            tmp_DF = pd.DataFrame().from_dict(full_dict)            
            tmp_DF.to_csv(outfold + 'AS_scaled_WS_TEMP.csv')

            print(f'{icond} {imdl} {ifold}')

full_DF = pd.DataFrame().from_dict(full_dict)            
full_DF.to_csv(outfold + 'AS_scaled_WS.csv')

            
 
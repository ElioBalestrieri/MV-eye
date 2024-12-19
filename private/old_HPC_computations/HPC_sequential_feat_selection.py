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

#%% tools for classification & features selection

# functions for PCA/SVM
from sklearn.svm import SVC

# pipeline definer
from sklearn.metrics import balanced_accuracy_score

from sklearn.feature_selection import SequentialFeatureSelector

rbf_svm = SVC(C=10)

#%% models and conditions

ExpConds = ['ECEO', 'VS']
mdltypes = ['TimeFeats', 'FreqBands']

#%% non parallel part

# concatenate files between participant, after within-participant normalization
list_accs = []

for ThisExpCond in ExpConds:
        
    L_Xtrain, L_Xtest, PC_IDs = [], [], []
    for imdl in mdltypes:
    
        data_type = ThisExpCond + '_' + imdl
        fullX_train, fullX_test, Y_train, Y_test = cat_subjs_train_test(infold, strtsubj=0, endsubj=29, 
                                                                ftype=data_type, tanh_flag=True, 
                                                                compress_flag=True, pca_kept_var=.9)     
        L_Xtrain.append(fullX_train['PC_aggregate'])
        L_Xtest.append(fullX_test['PC_aggregate'])
        PC_IDs.append(fullX_train['list_PC_identifiers'])
        
    mergedMdls_train = np.concatenate(L_Xtrain, axis=1)
    mergedMdls_test = np.concatenate(L_Xtest, axis=1)
    merged_IDs = np.array(PC_IDs[0] + PC_IDs[1]) 
    
    # define selector
    SeqSel = SequentialFeatureSelector(rbf_svm, n_features_to_select='auto', 
                                        tol=.001, cv=5, n_jobs=5, scoring='balanced_accuracy')
    # fit selector on training
    SeqSel.fit(mergedMdls_train, Y_train)

    # transform both train and test based on selector
    red_Xtrain = SeqSel.transform(mergedMdls_train)
    red_Xtest = SeqSel.transform(mergedMdls_test)

    # final SVM
    fin_SVM = SVC(C=10)
    fin_SVM.fit(red_Xtrain, Y_train)
    out_preds = fin_SVM.predict(red_Xtest)
    list_accs.append(balanced_accuracy_score(Y_test, out_preds))
        
    # prepare output
    train_DF = pd.DataFrame(data=red_Xtrain, columns=merged_IDs[SeqSel.support_])
    test_DF = pd.DataFrame(data=red_Xtest, columns=merged_IDs[SeqSel.support_])
    labels_train = pd.DataFrame(data=Y_train, columns=['label'])
    labels_test = pd.DataFrame(data=Y_test, columns=['label'])
    
    # save out
    foutname_trainX = outfold + 'SeqFeatSel_' + ThisExpCond + '_trainX.csv'
    foutname_testX = outfold + 'SeqFeatSel_' + ThisExpCond + '_testX.csv'
    foutname_trainY = outfold + 'SeqFeatSel_' + ThisExpCond + '_trainY.csv'
    foutname_testY = outfold + 'SeqFeatSel_' + ThisExpCond + '_testY.csv'

    train_DF.to_csv(foutname_trainX)
    test_DF.to_csv(foutname_testX)
    labels_train.to_csv(foutname_trainY)
    labels_test.to_csv(foutname_testY)
    
# create mini df with the two accuracies and save as csv
accs_DF = pd.DataFrame(data=list_accs, index=ExpConds)
accs_DF.to_csv(outfold + 'SeqFeatSel_overall_test_accuracy.csv')


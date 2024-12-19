#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:14:20 2023

@author: balestrieri
"""


# notebook for 4 conds features benchmarking
import sys
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# from openTSNE import TSNE

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test, cat_subjs_train_test_ParcelScale

#input folder 
infold = '../STRG_computed_features/Mdl_comparison/'


strt_subj = 1; end_subj = 16


# load & cat files into a coherent, preprocessed train and test datasets
list_ExpConds = ['ECEO_TimeFeats', 'VS_TimeFeats']
fullX_train, fullX_test, Y_train_A, Y_test_A, subjID_trials_labels = cat_subjs_train_test(infold, strtsubj=strt_subj, endsubj=end_subj, 
                                                                    ftype=list_ExpConds, tanh_flag=True, 
                                                                    compress_flag=True)     
# try on parcel-defined scaling
X_train, X_test, Y_train, Y_test, subjID_train, subjID_test = cat_subjs_train_test_ParcelScale(infold, strtsubj=strt_subj, endsubj=end_subj, 
                                                                                                  ftype=list_ExpConds, tanh_flag=True)     

#%%

subjsP = np.array(subjID_train + subjID_test)
subjsA = np.array(subjID_trials_labels)

counttrials = np.empty((29, 2))

acc = strt_subj
for isubj in np.unique(subjsP):    
    counttrials[acc, 0] = np.sum(subjsP==isubj)
    acc+=1
    
acc = strt_subj
for isubj in np.unique(subjsA):    
    counttrials[acc, 1] = np.sum(subjsA==isubj)
    acc+=1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:36:41 2023

@author: balestrieri
"""

import dask

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.datasets import load_iris
from sklearn.svm import SVC

import numpy as np

from time import process_time, sleep

# from dask.distributed import Client

#%%
@dask.delayed
def run_feat_selector(X, Y):

    # rbf_svm = SVC(C=10) 
    # SeqSel = SequentialFeatureSelector(rbf_svm, n_features_to_select='auto', 
    #                                     tol=.01, n_jobs=5, cv=5)
    
    # SeqSel.fit(X, Y)
    
    sleep(1)
    
    feats_selected = 42
    
    return feats_selected


# def run_feat_selector_nonpar(X, Y):

#     rbf_svm = SVC(C=10) 
#     SeqSel = SequentialFeatureSelector(rbf_svm, n_features_to_select='auto', 
#                                        tol=.01, cv=5)
    
#     SeqSel.fit(X, Y)
    
#     feats_selected = SeqSel.get_support()
    
#     return feats_selected


#%%

nreps = 1000
nfeats = 100

X1, y = load_iris(return_X_y=True)
X1 = np.random.randn(nreps, nfeats)
X2 = np.random.randn(nreps, nfeats)
X3 = np.random.randn(nreps, nfeats)

y = np.random.randint(0, high=2, size=(nreps))


multi_X = np.arange(1000)


#%%

# client = Client()

# par_time_start = process_time()
# all_selections = []
# for X in multi_X:
    
#     this_sel = client.submit(run_feat_selector_nonpar, X, y)
#     # all_selections.append(this_sel)
    
# par_time_stop = process_time()

# print("parallel computation in seconds:", par_time_stop-par_time_start) 


#%% actually launch process
par_time_start = process_time()
all_selections = []
for X in multi_X:
    
    this_sel = run_feat_selector(X, y)
    all_selections.append(this_sel)
    
    
countExc = dask.compute(all_selections)
par_time_stop = process_time()-par_time_start

# print("parallel computation in seconds:", par_time_stop-par_time_start) 


# ser_time_start = process_time()
# ser_selections = []
# for X in multi_X:
    
#     this_sel = run_feat_selector_nonpar(X, y)
#     ser_selections.append(this_sel)
    
# ser_time_stop = process_time()
# print("serial computation in seconds:", ser_time_stop-ser_time_start) 



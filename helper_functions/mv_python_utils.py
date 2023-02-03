#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:33:37 2023

GrossLab Utils

Some useful, shared, python code

@author: common
"""

import scipy.io as sio
import numpy as np

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler


#%% load *mat structures
# full credit to "mergen" @ stackoverflow
# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries?noredirect=1&lq=1

def loadmat_struct(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#%% concatenate the features across subjects
    
def cat_subjs(infold, best_feats, strtsubj=0, endsubj=22):    

    # define subjects range
    range_subjs = np.arange(strtsubj, endsubj)
        
    # define common transformations
    remove_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    trim_outliers = RobustScaler()

    accsubj = 0
    for isubj in range_subjs:

        # load file & extraction
        fname = infold + f'{isubj+1:02d}' + '_feats.mat'
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
        
        Y_labels = F['Y']

        subj_dict = {}
        
        acc_feat = 0
        for ifeat in best_feats:
            
            this_val = F['single_feats'][ifeat]        
            this_val = remove_nan.fit_transform(this_val)
            this_val = trim_outliers.fit_transform(this_val)
            
            if acc_feat == 0:

                aggr_feat = np.copy(this_val)
                
            else:
                
                aggr_feat = np.concatenate((aggr_feat, this_val), axis=1)

            subj_dict.update({ifeat : this_val})
            
            acc_feat += 1
            
        subj_dict.update({'aggregate' : aggr_feat})
        
        if isubj == strtsubj:
            
            full_Y = Y_labels
            full_X = subj_dict

        else:
            
            full_Y = np.concatenate((full_Y, Y_labels), axis=0)

            for key in subj_dict:
                
                pre_feat = full_X[key]
                post_feat = subj_dict[key]
                full_X[key] = np.concatenate((pre_feat, post_feat), axis=0)
                
        accsubj +=1
        
    print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
          str(strtsubj+1) + ' to ' + str(endsubj+1))
                
    return full_X, full_Y
    




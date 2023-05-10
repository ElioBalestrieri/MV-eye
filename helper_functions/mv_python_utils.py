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

# 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#%% concatenate the features across subjects
    
def cat_subjs(infold, best_feats=None, strtsubj=0, endsubj=22, ftype='feats', 
              tanh_flag=False, compress_flag=False):

    # define subjects range
    range_subjs = np.arange(strtsubj, endsubj)
        
    # define common transformations
    remove_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    trim_outliers = RobustScaler()

    accsubj = 0; subjID_trials_labels = []
    for isubj in range_subjs:

        SUBJid = [f'ID_{isubj+1:02d}']            
        
        # load file & extraction
        fname = infold + f'{isubj+1:02d}_' + ftype + '.mat'
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
        
        # condition labels        
        Y_labels = F['Y']
        
        # original trial order before reshuffling   
        try:
            trl_order = F['trl_order']        
        except:
            trl_order=np.array([np.nan])

        # generate subject's trials labels
        # and append to the common list
        this_subj_IDlabels = SUBJid*len(Y_labels)
        subjID_trials_labels = subjID_trials_labels + this_subj_IDlabels

        if not best_feats:
            loop_feats = F['single_feats'].keys()
        else:
            loop_feats = best_feats

        subj_dict = {}
        
        acc_feat = 0
        for ifeat in loop_feats:
            
            this_val = F['single_feats'][ifeat]        
            this_val = remove_nan.fit_transform(this_val)
            this_val = trim_outliers.fit_transform(this_val)
            
            if tanh_flag:
                this_val = np.tanh(this_val)
            
            # compress?
            if compress_flag:
                
                this_val = np.float32(this_val)
            
            
            if acc_feat == 0:

                aggr_feat = np.copy(this_val)
                
            else:
                
                aggr_feat = np.concatenate((aggr_feat, this_val), axis=1)

            subj_dict.update({ifeat : this_val})
            
            acc_feat += 1
            
        subj_dict.update({'aggregate' : aggr_feat})
        
        if isubj == strtsubj:
            
            full_Y = Y_labels
            full_trl_order = trl_order
            full_X = subj_dict

        else:
            
            full_Y = np.concatenate((full_Y, Y_labels), axis=0)
            full_trl_order = np.concatenate((full_trl_order, trl_order), axis=0)

            for key in subj_dict:
                
                pre_feat = full_X[key]
                post_feat = subj_dict[key]
                full_X[key] = np.concatenate((pre_feat, post_feat), axis=0)
                
        accsubj +=1
        
    print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
          str(strtsubj+1) + ' to ' + str(endsubj+1))
                
    return full_X, full_Y, subjID_trials_labels, full_trl_order
    

#%% concatenate subjects but while concurrently splitting training and test

def cat_subjs_train_test(infold, best_feats=None, strtsubj=0, endsubj=22, ftype='feats', 
                         tanh_flag=False, compress_flag=False, test_size=.15, pca_kept_var=None):

    # define subjects range
    range_subjs = np.arange(strtsubj, endsubj)
        
    # define common transformations
    remove_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    trim_outliers = RobustScaler()

    accsubj = 1; subjID_trials_labels = []
    for isubj in range_subjs:

        SUBJid = [f'ID_{isubj+1:02d}']            
        
        # allow concatenation of multiple file types, aka experimental conditions
        # in order to then run decoding across conds
        if isinstance(ftype, str):            
            ftype = [ftype]
                    
        acc_label = 0; tmp_dict = {}
        for this_ftype in ftype:
            
            # load file & extraction
            fname = infold + f'{isubj+1:02d}_' + this_ftype + '.mat'
            mat_content = loadmat_struct(fname)
            F = mat_content['variableName']
            
            if not best_feats:
                loop_feats = F['single_feats'].keys()
            else:
                loop_feats = best_feats

            # condition labels        
            Y_labels = F['Y']+acc_label
            
            for ifeat in loop_feats:
                     
                if acc_label == 0:                    
                    tmp_dict.update({ifeat : np.copy(F['single_feats'][ifeat])})

                else:
                    tmp_dict[ifeat] = np.concatenate((tmp_dict[ifeat], np.copy(F['single_feats'][ifeat])), axis=0)

            if acc_label == 0:
                swap_Y = np.copy(Y_labels)
            else:
                swap_Y = np.concatenate((swap_Y, Y_labels), axis=0)

            acc_label+=len(np.unique(Y_labels))
            
        # assign the loop output to the original structures
        F['single_feats'] = tmp_dict
        Y_labels = swap_Y

        # get indexes for train and test
        idx_full = np.arange(len(Y_labels))
        idx_train, idx_test = train_test_split(idx_full, test_size=test_size, random_state=isubj)

        # labels split in train and test
        Y_train = Y_labels[idx_train]; Y_test = Y_labels[idx_test]

        # generate subject's trials labels
        # and append to the common list
        this_subj_IDlabels = SUBJid*len(Y_labels)
        subjID_trials_labels = subjID_trials_labels + this_subj_IDlabels

        if not best_feats:
            loop_feats = F['single_feats'].keys()
        else:
            loop_feats = best_feats

        subj_dict_train, subj_dict_test = {}, {}
        
        acc_feat = 0
        for ifeat in loop_feats:
            
            X = F['single_feats'][ifeat]     
            X_train = X[idx_train, :]; X_test = X[idx_test, :]
            
            X_train = remove_nan.fit_transform(X_train)
            X_test = remove_nan.fit_transform(X_test)
            
            # fit robust scaler on xtrain and apply it on x test
            rob_scal_mdl = trim_outliers.fit(X_train)
            X_train = rob_scal_mdl.transform(X_train)
            X_test = rob_scal_mdl.transform(X_test)
                    
            if tanh_flag:

                X_train = np.tanh(X_train)
                X_test = np.tanh(X_test)

            # compress?
            if compress_flag:
                
                X_train = np.float32(X_train)
                X_test = np.float32(X_test)
                    
            if acc_feat == 0:

                aggregate_train = np.copy(X_train)
                aggregate_test = np.copy(X_test)
            
            else:
                
                aggregate_train = np.concatenate((aggregate_train, 
                                                  X_train), axis=1)

                aggregate_test = np.concatenate((aggregate_test, 
                                                 X_test), axis=1)

            subj_dict_train.update({ifeat : X_train})
            subj_dict_test.update({ifeat : X_test})

            acc_feat += 1
            
        subj_dict_train.update({'aggregate' : aggregate_train})
        subj_dict_test.update({'aggregate' : aggregate_test})

        
        if isubj == strtsubj:
            
            #initialize datasets that will be concatenated
            full_Y_train = Y_train; full_Y_test = Y_test   
            full_X_train = subj_dict_train
            full_X_test = subj_dict_test
            
        else:
            
            full_Y_train = np.concatenate((full_Y_train, Y_train), axis=0)
            full_Y_test = np.concatenate((full_Y_test, Y_test), axis=0)

            for key in subj_dict_train:
                
                pre_feat_train = full_X_train[key]
                pre_feat_test = full_X_test[key]

                post_feat_train = subj_dict_train[key]
                post_feat_test = subj_dict_test[key]

                full_X_train[key] = np.concatenate((pre_feat_train, 
                                                    post_feat_train), axis=0)

                full_X_test[key] = np.concatenate((pre_feat_test, 
                                                    post_feat_test), axis=0)

                
        accsubj +=1
        
    print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
          str(strtsubj+1) + ' to ' + str(endsubj+1))

    if pca_kept_var:
                
        acc_feat = 0; list_PC_identifiers = []
        for key in loop_feats: # call the loop feats, so that "aggregate" is not included. we create a new aggregate by concatenating the PCs
             
            my_PCA = PCA(n_components=.9, svd_solver='full')
        
            X_train = full_X_train[key]
            X_test = full_X_test[key]

            my_PCA = my_PCA.fit(X_train)
            
            PC_Xtrain = my_PCA.transform(X_train)
            PC_Xtest = my_PCA.transform(X_test)
            
            # get shape to assign feats IDs
            for PCidx in range(PC_Xtrain.shape[1]):
                
                PCnum=f'{PCidx+1:02d}'
                list_PC_identifiers.append(key + '_PC' + PCnum)
                           
            full_X_train.update({'PC_' + key : PC_Xtrain})
            full_X_test.update({'PC_' + key : PC_Xtest})

            if acc_feat == 0:

                aggregate_train = np.copy(PC_Xtrain)
                aggregate_test = np.copy(PC_Xtest)

            else:
                
                aggregate_train = np.concatenate((aggregate_train, 
                                                  PC_Xtrain), axis=1)

                aggregate_test = np.concatenate((aggregate_test, 
                                                 PC_Xtest), axis=1)

            acc_feat += 1
        
        full_X_train.update({'PC_aggregate' : aggregate_train})
        full_X_test.update({'PC_aggregate' : aggregate_test})

        full_X_train.update({'list_PC_identifiers' : list_PC_identifiers})
        full_X_test.update({'list_PC_identifiers' : list_PC_identifiers})
        

    return full_X_train, full_X_test, full_Y_train, full_Y_test
















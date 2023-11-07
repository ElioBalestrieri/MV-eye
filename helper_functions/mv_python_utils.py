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
import copy

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler

# 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import pickle



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
    
def cat_subjs(infold, best_feats=None, subjlist=None, strtsubj=0, endsubj=22, ftype='feats', 
              tanh_flag=False, compress_flag=False, all_feats_flag=True):

    # define subjects range
    if subjlist is None:
        range_subjs = np.arange(strtsubj, endsubj)
    else:
        range_subjs = subjlist
        strtsubj = subjlist[0]
        endsubj = subjlist[-1]
        
    # define common transformations
    remove_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    trim_outliers = RobustScaler()

    accsubj = 0; subjID_trials_labels = []
    for isubj in range_subjs:

        if isinstance(isubj, np.integer):
            SUBJid = [f'ID_{isubj+1:02d}']            
        else:
            SUBJid = [isubj]
                
        # load file & extraction
        if isinstance(isubj, np.integer):
            fname = infold + f'{isubj+1:02d}_' + ftype + '.mat'                
        else:
            fname = infold + isubj + '_' + ftype + '.mat'


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

            if all_feats_flag:
                subj_dict.update({ifeat : this_val})

            acc_feat += 1
            
        subj_dict.update({'full_set' : aggr_feat})
        
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
        
    if isinstance(isubj, np.integer):        
        print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
              str(strtsubj+1) + ' to ' + str(endsubj+1))
    else:
        print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
              strtsubj + ' to ' + endsubj)
                
    return full_X, full_Y, subjID_trials_labels, full_trl_order
    

#%% concatenate subjects but while concurrently splitting training and test

def cat_subjs_train_test(infold, best_feats=None, subjlist=None, strtsubj=None, 
                         endsubj=None, ftype='feats', tanh_flag=False, 
                         compress_flag=False, test_size=.15, pca_kept_var=None):

    # define subjects range
    if subjlist is None:
        range_subjs = np.arange(strtsubj, endsubj)
    else:
        range_subjs = subjlist
        strtsubj = subjlist[0]
        endsubj = subjlist[-1]

        
    # define common transformations
    remove_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    trim_outliers = RobustScaler()

    accsubj = 1; subjID_trials_labels = []
    for isubj in range_subjs:

        if isinstance(isubj, np.integer):
            SUBJid = [f'ID_{isubj+1:02d}']            
        else:
            SUBJid = [isubj]
        
        # allow concatenation of multiple file types, aka experimental conditions
        # in order to then run decoding across conds
        if isinstance(ftype, str):            
            ftype = [ftype]
                    
        acc_label = 0; tmp_dict = {}
        for this_ftype in ftype:
            
            # load file & extraction
            if isinstance(isubj, np.integer):
                fname = infold + f'{isubj+1:02d}_' + this_ftype + '.mat'                
            else:
                fname = infold + isubj + '_' + this_ftype + '.mat'


                
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
        F['single_feats'] = copy.deepcopy(tmp_dict)
        Y_labels = copy.deepcopy(swap_Y)

        # get indexes for train and test
        idx_full = np.arange(len(Y_labels))
        if isinstance(isubj, np.integer):        
            idx_train, idx_test = train_test_split(idx_full, test_size=test_size, 
                                                   random_state=isubj)
        else:
            idx_train, idx_test = train_test_split(idx_full, test_size=test_size, 
                                                   random_state=accsubj)

            
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
            
        subj_dict_train.update({'full_set' : aggregate_train})
        subj_dict_test.update({'full_set' : aggregate_test})

        
        if isubj == strtsubj:
            
            #initialize datasets that will be concatenated
            full_Y_train = Y_train
            full_Y_test = Y_test   
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

    if isinstance(isubj, np.integer):        
        print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
              str(strtsubj+1) + ' to ' + str(endsubj+1))
    else:
        print('Concatenated ' + str(accsubj) + ' subjects, from ' + 
              strtsubj + ' to ' + endsubj)
        

    if pca_kept_var:
                
        acc_feat = 0; list_PC_identifiers = []
        for key in loop_feats: # call the loop feats, so that "full_set" is not included. we create a new aggregate by concatenating the PCs
             
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
        
        full_X_train.update({'PC_full_set' : aggregate_train})
        full_X_test.update({'PC_full_set' : aggregate_test})

        full_X_train.update({'list_PC_identifiers' : list_PC_identifiers})
        full_X_test.update({'list_PC_identifiers' : list_PC_identifiers})
        

    return full_X_train, full_X_test, full_Y_train, full_Y_test, subjID_trials_labels 




#%% concatenate subjects but while concurrently splitting training and test

def first_dat_preproc(X, remove_nan, trim_outliers, tanh_flag=False, 
                      compress_flag=False):
    
    X = remove_nan.fit_transform(X)
    
    # fit robust scaler on xtrain and apply it on x test
    X = trim_outliers.fit_transform(X)
            
    if tanh_flag:
        X = np.tanh(X)
    
    # compress?
    if compress_flag:        
        X = np.float32(X)

    return X



def cat_subjs_from_list(subjs_list, infold, ftype, tanh_flag, compress_flag):

    # preallocate dictionaries
    subjs_dict = {}

    # define common transformations
    remove_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    trim_outliers = RobustScaler()

    accsubj = 1; subjID_trials_labels = []      
    for isubj in subjs_list:

        if isinstance(isubj, np.integer):
            SUBJid = [f'ID_{isubj+1:02d}']            
        else:
            SUBJid = [isubj]
    
                            
        # load file & extraction
        if isinstance(isubj, np.integer):
            fname = infold + f'{isubj+1:02d}_' + ftype + '.mat'                
        else:
            fname = infold + isubj + '_' + ftype + '.mat'

            
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
        
        loop_feats = list(F['single_feats'].keys())

        # condition labels        
        Y = F['Y']
        
        # generate subject's trials labels
        # and append to the common list
        this_subj_IDlabels = SUBJid*len(Y)
        subjID_trials_labels = subjID_trials_labels + this_subj_IDlabels

        for ifeat in loop_feats:
            
            X = F['single_feats'][ifeat]     
            
            X = first_dat_preproc(X, remove_nan, trim_outliers, 
                                        tanh_flag=tanh_flag, 
                                        compress_flag=compress_flag)
                
            if accsubj==1:                
                subjs_dict.update({ifeat : X})
            else:
                subjs_dict[ifeat] = np.concatenate((subjs_dict[ifeat], X), axis=0)

        # now concatenate y labels                
        if accsubj==1:
            full_Y = Y
        else:
            full_Y = np.concatenate((full_Y, Y), axis=0)
                
                
        accsubj +=1

    return subjs_dict, full_Y, subjID_trials_labels



def split_subjs_train_test(infold, partition_dict, nfolds=5, ftype='feats', tanh_flag=False, 
                         compress_flag=False, pca_kept_var=None):

    out_par_conds = []
    for ifold in range(nfolds):
        
        # get subjlist for training 
        subjs_trainset = partition_dict['train_fold_' + str(ifold)]

        # get subjlist for testing 
        subjs_testset = partition_dict['test_fold_' + str(ifold)]

        # apply basic preprcoessing for training... 
        X_train, Y_train, subjIDtrain = cat_subjs_from_list(subjs_trainset, infold, 
                                                            ftype, tanh_flag, compress_flag)
        # ... & testing separately
        X_test, Y_test, subjIDtest = cat_subjs_from_list(subjs_testset, infold, 
                                                         ftype, tanh_flag, compress_flag)
        loop_feats = list(X_train.keys())     
        
        acc_feat = 0; list_PC_identifiers = []            
        for key in loop_feats: # call the loop feats, so that "full_set" is not included. we create a new aggregate by concatenating the PCs

            X_train_single_feat = X_train[key]
            X_test_single_feat = X_test[key]

            if pca_kept_var:

                # apply PCA for each single feature, and concatenate. To avoid confounders, 
                # while retaining coherence between components the PCA is fitted on the 
                # train, and the same model is then applied to both train and test
         
                my_PCA = PCA(n_components=pca_kept_var, svd_solver='full')
                my_PCA = my_PCA.fit(X_train_single_feat)
        
                X_train_single_feat = my_PCA.transform(X_train_single_feat)
                X_test_single_feat = my_PCA.transform(X_test_single_feat)

            if acc_feat == 0 :                    
                Xtrain_full = X_train_single_feat
                Xtest_full = X_test_single_feat
            else:
                Xtrain_full = np.concatenate((Xtrain_full, X_train_single_feat), axis=1)
                Xtest_full = np.concatenate((Xtest_full, X_test_single_feat), axis=1)
            
            acc_feat += 1
        
        # parallel condition: one CPU will load this Fold/ftype combination and
        # work on that
        this_par_cond = 'Fold_' + str(ifold) + '_' + ftype
        
        # save output 
        current_out = {this_par_cond + '_X_train': Xtrain_full,
                       this_par_cond + '_X_test' : Xtest_full,
                       this_par_cond + '_Y_train': Y_train,
                       this_par_cond + '_Y_test': Y_test,
                       this_par_cond + '_SubjIDtrain': subjIDtrain,
                       this_par_cond + '_SubjIDtest': subjIDtest}
        
        for key, val in current_out.items():
            
            fname = infold + key + '.pickle'
            with open(fname, 'wb') as handle:
                pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # append parallel condition. The final list is function output
        out_par_conds.append(this_par_cond)
            
    return out_par_conds


def cat_subjs_train_test_ParcelScale(infold, best_feats=None, subjlist=None, strtsubj=None, 
                         endsubj=None, ftype='feats', tanh_flag=False, 
                         test_size=.15):

    # define subjects range
    if subjlist is None:
        range_subjs = np.arange(strtsubj, endsubj)
    else:
        range_subjs = subjlist
        strtsubj = subjlist[0]
        endsubj = subjlist[-1]

    # define common transformations
    trim_outliers = RobustScaler()

    accsubj = 1; 
    X_train, X_test = {}, {}
    for isubj in range_subjs:

        
        # allow concatenation of multiple file types, aka experimental conditions
        # in order to then run decoding across conds
        if isinstance(ftype, str):            
            ftype = [ftype]
                    
        acc_label = 0; 
        for this_ftype in ftype:
            
            # load file & extraction
            if isinstance(isubj, np.integer):
                fname = infold + f'{isubj+1:02d}_' + this_ftype + '.mat'                
            else:
                fname = infold + isubj + '_' + this_ftype + '.mat'
                
            mat_content = loadmat_struct(fname)
            F = mat_content['variableName']
            
            if not best_feats:
                loop_feats = F['single_feats'].keys()
            else:
                loop_feats = best_feats

            # condition labels        
            Y_labels = F['Y']+acc_label
            
            # get indexes for train and test
            idx_full = np.arange(len(Y_labels))
        
            if isinstance(isubj, np.integer):        
                idx_train, idx_test = train_test_split(idx_full, test_size=test_size, 
                                                       random_state=isubj)
            else:
                idx_train, idx_test = train_test_split(idx_full, test_size=test_size, 
                                                       random_state=accsubj)

            # generate subject's trials labels
            # and append to the common list
            IDlabels_train = [isubj]*len(idx_train)
            IDlabels_test = [isubj]*len(idx_test)

                
            # create labels & subjID vectors for train and test
            if (isubj == strtsubj) & (acc_label==0):
                Y_train = Y_labels[idx_train]; Y_test = Y_labels[idx_test]      
                IDs_allsubjs_train = IDlabels_train
                IDs_allsubjs_test = IDlabels_test
            
            else:            
                Y_train = np.concatenate((Y_train,  Y_labels[idx_train]), axis=0)
                Y_test = np.concatenate((Y_test, Y_labels[idx_test]), axis=0)
                IDs_allsubjs_train = IDs_allsubjs_train + IDlabels_train
                IDs_allsubjs_test = IDs_allsubjs_test + IDlabels_test
            
            for ifeat in loop_feats:

                if ifeat == 'fooof_aperiodic':
                    
                    tmp_dat = [F['single_feats'][ifeat][:, 0::2], F['single_feats'][ifeat][:, 0::2]]
                    featname = ['fooof_slope', 'fooof_offset']

                else:
                
                    tmp_dat = [F['single_feats'][ifeat]] # all of this superconvoluted nested loop that follows is 
                                                         # to account that fooof contains 2 fu**ng feats instead of one
                    featname = [ifeat]
                
                acc_this_feat = 0
                for this_feat in featname:

                    dat = tmp_dat[acc_this_feat]
                    scaled_dat = trim_outliers.fit_transform(dat.T).T # apply transformation on parcels, and transpose it back
                
                    if tanh_flag:
                        X = np.tanh(scaled_dat)

                    if (isubj == strtsubj) & (acc_label==0):                    
                        X_train.update({this_feat : X[idx_train, :]})
                        X_test.update({this_feat : X[idx_test, :]})

                    else:
                        X_train[this_feat] = np.concatenate((X_train[this_feat], X[idx_train, :]), axis=0)
                        X_test[this_feat] = np.concatenate((X_test[this_feat], X[idx_test, :]), axis=0)

                        
                    acc_this_feat += 1
                    
            acc_label+=len(np.unique(Y_labels))
            
    # finally, out
    return X_train, X_test, Y_train, Y_test, IDs_allsubjs_train, IDs_allsubjs_test
# -*- coding: utf-8 -*-
"""
two purposes of the code:
1. preprocess data by reading files, concatenating them, and storing them on disk
2. launch parallel classifier on features in loop for filetype (freqtype)
"""

# general import
# 1st part
import sys
import pickle

# 2nd part
# system & basic libraries
import numpy as np
import pandas as pd
import dask
import os
# from datetime import datetime
# classification tools
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
# crossvalidation
from sklearn.model_selection import cross_val_score
# pipeline definer
from sklearn.pipeline import Pipeline
# imputer
from sklearn.impute import SimpleImputer
# scaler
from sklearn.preprocessing import RobustScaler
# shuffler (before SVC, for mitigating across subjects differences in training?)
from sklearn.utils import shuffle

############################################ 1st part ##################################################################

# setting path for mv_python_utils
sys.path.append('../helper_functions/')
from mv_python_utils import cat_subjs

# define fre bands as file strings
freqbands = ['delta_1_4_Hz', 'theta_4_8_Hz', 'alpha_8_13_Hz', 'beta_13_30_Hz',
             'low_gamma_30_45_Hz', 'high_gamma_55_100_Hz', 'feats']

# input folder
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# best feats (only for local testing)
best_feats = ['MCL', 'std']

for iBPcond in freqbands:

    # generate datasets
    full_train_X, full_train_Y, foo = cat_subjs(infold, best_feats=best_feats, strtsubj=0, endsubj=22, ftype=iBPcond)
    full_test_X, full_test_Y, foo = cat_subjs(infold, best_feats=best_feats, strtsubj=22, endsubj=29, ftype=iBPcond)

    list_dsets = [full_train_X, full_train_Y, full_test_X, full_test_Y]
    set_names = ['train', 'test']; set_types = ['X', 'Y']

    set_acc = 0
    for iname in set_names:
        for itype in set_types:
            fname = infold + iname + '_' + itype + '_merged_' + iBPcond + '.pickle'

            with open(fname, 'wb') as handle:
                pickle.dump(list_dsets[set_acc], handle, protocol=pickle.HIGHEST_PROTOCOL)

            set_acc += 1


############################################## 2nd part (parallel) #####################################################


# define the parallel function to be called in separate threads
@dask.delayed
def par_classify(X_train, Y_train, X_test, Y_test, featname):

    # give feedback on process started
    print(featname + ' is getting computed')

    # define the classification pipeline
    # definition necessary here to avoid weird overhead bugs, where models of one feature were tested (or at least,
    # prepared in size preallocation) for a parallel feature
    pipe = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, strategy='mean')),
                     ('scaler', RobustScaler()),
                     ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                     ('SVM', SVC(C=10))
                    ])

    # shuffle train and test
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

    # first obtain crossvalidated accuracy on the training test
    try:
        CV_acc = cross_val_score(pipe, X_train, Y_train, cv=10,
                              scoring='balanced_accuracy').mean()
    except:
        CV_acc = np.nan


    try:
        # train full model
        full_mdl = pipe.fit(X_train, Y_train)

        # generalize and evaluate generalization accuracy
        test_preds_Y = full_mdl.predict(X_test)
        GEN_acc = balanced_accuracy_score(Y_test, test_preds_Y)

    except:

        foo = 'moo'
        GEN_acc = np.nan

    # return a DF: col>feat name; rows>CV_accuracy, GEN_accuracy
    rownames = ['CV_accuracy', 'GEN_accuracy']
    DF = pd.DataFrame(data=[CV_acc, GEN_acc], columns=[featname], index=rownames)

    return DF

######## Loop across freqbands

for iBPcond in freqbands:

    fname_Xtrain = infold + 'train_X_merged_' + iBPcond + '.pickle'
    fname_Ytrain = infold + 'train_Y_merged_' + iBPcond + '.pickle'
    fname_Xtest = infold + 'test_X_merged_' + iBPcond + '.pickle'
    fname_Ytest = infold + 'test_Y_merged_' + iBPcond + '.pickle'

    with open(fname_Xtrain, 'rb') as handle:
        X_train = pickle.load(handle)
    with open(fname_Ytrain, 'rb') as handle:
        Y_train = pickle.load(handle)
    with open(fname_Xtest, 'rb') as handle:
        X_test = pickle.load(handle)
    with open(fname_Ytest, 'rb') as handle:
        Y_test = pickle.load(handle)

    list_DF = []

    for key in X_train:

        X_ = X_train[key]
        X_leftout = X_test[key]

        DF = par_classify(X_, Y_train, X_leftout, Y_test, key)
        list_DF.append(DF)

    # actually launch the process
    out_DFs = dask.compute(list_DF)
    ordered_accs_DF = pd.concat(out_DFs[0], axis=1)

    # save
    fname_out = '../STRG_decoding_accuracy/' + iBPcond + '_intersubjs_accs.csv'
    ordered_accs_DF.to_csv(fname_out)


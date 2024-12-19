# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os


#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct


#%% Pipeline object definition

# import the main elements needed
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# crossvalidation
from sklearn.model_selection import cross_val_score

# pipeline definer
from sklearn.pipeline import Pipeline

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler # to be used at some point?

from sklearn.preprocessing import FunctionTransformer


# deinfine the pipeline to be used 



class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
                                                            strategy='mean')),
                           ('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10))
                           ])

#%% folder(s) definition

#input folder 
infold = '../STRG_data/reconstruct_signal/'

# # output folder
# outfold = '../STRG_decoding_accuracy/'
# if not(os.path.isdir(outfold)):
#     os.mkdir(outfold)


fname = '01_ECEO_TimeFeats.mat'


mat_content = loadmat_struct(infold+fname)



F = mat_content['variableName']
test_ratio = .1 # for the example of 10 fold cv
trl_order = F['trl_order']

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import LeaveOneGroupOut

discr =  KBinsDiscretizer(n_bins=int(1/test_ratio), encode='ordinal', strategy='uniform')
A = discr.fit_transform(trl_order[:, np.newaxis])

logo = LeaveOneGroupOut()
logo.get_n_splits(groups=A)

#%%


Y = F['Y']
single_feats = F['single_feats']

this_feat = 'MCL'
X = single_feats[this_feat]


acc = cross_val_score(class_pipeline, X, Y, cv=logo.get_n_splits(groups=A), 
                      scoring='balanced_accuracy')








# basic imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import glob
import re
import os

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

# allow parallelization
import dask
# import all packages needed for classification
from sklearn.svm import SVC
# from sklearn.metrics import balanced_accuracy_score # not needed now, called internally from pipeline object
# functions for PCA
from sklearn.decomposition import PCA
# pipeline definer
from sklearn.pipeline import Pipeline
# scaler
from sklearn.preprocessing import RobustScaler
# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer
# crossvalidation
from sklearn.model_selection import cross_val_score

# path definition: list all the files matching patterns to be piped into the parallel function
path = '../STRG_computed_features/alphaSPADE/' + '*_timeresolved.mat'
filelist = glob.glob(path)

# output folder
outfold = '../STRG_decoding_accuracy/alphaSPADE/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# define the pipeline to be used
class_pipeline = Pipeline([('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10, random_state=42))
                           ])

########### function definition 
@dask.delayed
def single_subj_classify(fname, outfold, class_pipeline, cv_fold=5):

    # load stuff
    mat_content = loadmat_struct(fname)
    F = mat_content['variableName']
    
    # extract subject ID from filename
    regex = re.compile(r'\d+')
    subjID = regex.findall(fname)[0]
    
    # define fname for output saving
    out_fname = outfold + subjID + '_timeresolved_acc.csv'

    # preamble
    mask_evs = F['trialinfo'][:,1]>=10 # erase catch, fttb
    Y = F['trialinfo'][mask_evs,1] # 6 classes classification
    y_H_M = (Y==11) | (Y==22)
    randY = copy.deepcopy(y_H_M)
    np.random.shuffle(randY)
    
    dict_accs = {}
    for ifeat in F['single_feats'].keys():

        ntpoints = F['single_feats'][ifeat].shape[1]    
        dict_accs[ifeat] = []

        for iT in range(ntpoints):

            X = F['single_feats'][ifeat][:, iT, mask_evs].T

            acc = cross_val_score(class_pipeline, X, y_H_M, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
            rand_acc = cross_val_score(class_pipeline, X, randY, cv=cv_fold, 
                                        scoring='balanced_accuracy').mean()

            dict_accs[ifeat].append(acc-rand_acc)

        print(ifeat + ': ' + str(iT) + '/' + str(ntpoints))
    
    # finalize dataframe
    dict_accs['time (s)'] = F['time_winCENter']
    DF_accs = pd.DataFrame.from_dict(dict_accs)
    DF_accs = DF_accs.set_index('time (s)')

    DF_accs.to_csv(fname_out)
    print(subjID)   
    
    return(subjID)


# loop pipeline 
allsubjs = []
for ifname in filelist:

    thisubjs = single_subj_classify(ifname, outfold, class_pipeline)
    allsubjs.append(thisubj)
    
# actually launch process
dask.compute(allsubjs)




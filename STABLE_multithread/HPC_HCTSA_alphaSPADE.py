#!/usr/bin/env python
# coding: utf-8

# In[1]:


# basic imports
import numpy as np
import pandas as pd
import h5py
import copy
import os
import glob
import dask

# import all packages needed for classification
from sklearn.svm import SVC
# from sklearn.metrics import balanced_accuracy_score # not needed now, called internally from pipeline object
# functions for PCA
from sklearn.decomposition import PCA
# pipeline definer
from sklearn.pipeline import Pipeline
# trim bad values
from sklearn.impute import SimpleImputer
# scaler
from sklearn.preprocessing import RobustScaler
# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer
# crossvalidation
from sklearn.model_selection import cross_val_score

# define the pipeline to be used
class_pipeline = Pipeline([('trimmer', SimpleImputer(missing_values=np.nan, strategy='median')),
                           ('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(kernel='linear', random_state=42))
                           ])


# In[2]: preamble

# define input folder and current subject. 
infold = '../STRG_computed_features/alphaSPADE/HCTSA_prestim/'
# output folder
outfold = '../STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim/'
if not(os.path.isdir(outfold)):
    os.makedirs(outfold)
    
filelist = glob.glob(infold+'*.h5')

subjlist = []
for ifile in filelist:
    
    subjlist.append(ifile[-7:-3])

# In[3]: parallel function definition

@dask.delayed
def decode_HCTSA(infold, this_subj, class_pipeline, cv_fold=5):

    # read file
    f = h5py.File(infold + this_subj + '.h5', 'r')
    
    # fetch targets
    Y = np.array(f['Y'])[0, :]
    
    # create shuffled vector of labels
    randY = copy.deepcopy(Y)
    np.random.shuffle(randY)
    
    # initialize DF
    dict_accs = {}
    
    acc_feat = 0
    for featname, tmpX in f['X'].items():
            
        X = np.array(tmpX)
        
        try:
        
            # substitue infinite values (if any). Those will be substituted with the 
            # median in Pipeline (separately for train and test)
            X[np.abs(X)>1e308] = np.nan
            
            acc = cross_val_score(class_pipeline, X, Y, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
            rand_acc = cross_val_score(class_pipeline, X, randY, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
            
            delta_acc = acc-rand_acc

        except:
            
            delta_acc = np.nan
        
        dict_accs[featname] = [delta_acc]
        
        # partial data storage
        if np.mod(acc_feat, 100)==0:
            
            DF_accs = pd.DataFrame.from_dict(dict_accs)
            DF_accs.to_csv(outfold + this_subj + '.csv')
            
        acc_feat += 1

    # store full data at the end
    DF_accs = pd.DataFrame.from_dict(dict_accs)
    DF_accs.to_csv(outfold + this_subj + '.csv')
        
    return this_subj

    
# In[4]: launch parallel computation

# loop pipeline 
allsubjs = []
for isubj in subjlist:

    thisubj = decode_HCTSA(isubj, outfold, class_pipeline)
    allsubjs.append(thisubj)
    
# actually launch process
dask.compute(allsubjs)


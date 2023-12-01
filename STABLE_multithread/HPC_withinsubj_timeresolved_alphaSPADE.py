# basic imports
import sys
import numpy as np
import pandas as pd
import copy
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

# path & subjIDs definition
path = '../STRG_computed_features/alphaSPADE/timeresolved_split_feats/'
subjIDs = os.listdir(path); subjlist = []
for ID in subjIDs:
    subjlist.append(path+ID+'/*.mat')

# output folder
outfold = '../STRG_decoding_accuracy/alphaSPADE/timeresolved/'
if not(os.path.isdir(outfold)):
    os.makedirs(outfold)

# define the pipeline to be used
class_pipeline = Pipeline([('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10, random_state=42))
                           ])


########### parallel function definition 
@dask.delayed
def single_subj_classify(fpath, outfold, class_pipeline, cv_fold=5):

       
    # extract subject ID from filename
    regex = re.compile(r'\d+')
    subjID = regex.findall(fpath)[0]
    
    # each subj needs an output folder
    subjspec_outfold = outfold + subjID + '/'
    if not(os.path.isdir(subjspec_outfold)):
        os.makedirs(subjspec_outfold)

    
    hdr_fname = fpath + subjID + '_HDR.mat'
    HDR = loadmat_struct(hdr_fname)['HDR']
    
    featnames = list(HDR['featnames'])
    
    for thisfeat in featnames:
        
        tmp_fname = fpath + subjID + '_' + thisfeat + '.mat'
        D = loadmat_struct(tmp_fname)['Decode']
    
        mask_evs = D['rawY'][:,1]>=10 # erase catch, fttb
        Y = D['rawY'][mask_evs,1] # 6 classes classification
        y_H_M = (Y==11) | (Y==22)
        randY = copy.deepcopy(y_H_M)
        np.random.shuffle(randY)
    
        Xovertime = D['X'][mask_evs, :, :]
        ntpoints = Xovertime.shape[1]    
        
        dict_accs = {'time_winCENter' : HDR['time_winCENter'],
                     'time_winOFFset' : HDR['time_winOFFset'],
                     'time_winONset' : HDR['time_winONset'],
                     'accuracy' : [],
                     'delta_accuracy' : []}
        
        for iT in range(ntpoints):
    
            X = Xovertime[:, iT, :]
            acc = cross_val_score(class_pipeline, X, y_H_M, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
            rand_acc = cross_val_score(class_pipeline, X, randY, cv=cv_fold, 
                                        scoring='balanced_accuracy').mean()
    
            dict_accs['accuracy'].append(acc)
            dict_accs['delta_accuracy'].append(acc-rand_acc)
            print(iT)
            
        # define fname for output saving
        out_fname = subjspec_outfold + subjID + '_' + thisfeat +'.csv'
        
        # save Dataframe as csv
        DF_accs = pd.DataFrame.from_dict(dict_accs)
        DF_accs.to_csv(out_fname)
        
        print(subjID)   
    
    return(subjID)


# loop pipeline 
allsubjs = []
for ifname in subjlist:

    thisubjs = single_subj_classify(ifname, outfold, class_pipeline)
    allsubjs.append(thisubjs)
    
# actually launch process
dask.compute(allsubjs)




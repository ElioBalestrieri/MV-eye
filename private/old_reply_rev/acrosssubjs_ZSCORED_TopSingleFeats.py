# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
np.random.seed(42)
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os
import copy
# import dask
from datetime import datetime

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# pipeline definer
from sklearn.pipeline import Pipeline

# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer

# crossvalidation
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer # to be used at some point?
from sklearn import metrics

# various 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import kl_div


#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/rev_reply_MEG_zscore/'

# output folder
outfold = '../STRG_decoding_accuracy/rev_reply_MEG_zscore/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

#%%

expconds = ['VS', 'ECEO'];
mdltypes = ['TimeFeats', 'FreqBands']

topfeats_cond_names = {'ECEO' : {'TimeFeats' : ['CO_Embed2_Dist_tau_d_expfit_meandiff', 'SB_MotifThree_quantile_hh'],
                                 'FreqBands' : []},
                       'VS' : {'TimeFeats': ['CO_f1ecac'],
                               'FreqBands' :['high_gamma_power_55_100_Hz']}
                       }

#%%    

full_subj_dict = {}

for icond in expconds:
                    
    full_subj_dict.update({icond : {'X':[], 'Y': np.empty((0,1), dtype=np.uint8), 'groups':[], 'single features' : {}}})

    acc_mdl = 0
    for imdl in mdltypes:

        featnames = topfeats_cond_names[icond][imdl]        

        acc_feat = 0 
        for ifeat in featnames:
                
            full_subj_dict[icond]['single features'].update({ifeat : []})     
                
            if (ifeat == 'mean') or (ifeat == 'std'):
                    
                continue
        
            for isubj in range(29):
                
                fname = infold + f'{isubj+1:02d}_{icond}_{imdl}.mat'
    
                mat_content = loadmat_struct(fname)
                F = mat_content['variableName']
                Y_labels = F['Y']
                single_feats = F['single_feats']        

                sclr = RobustScaler()
                scaled_X = sclr.fit_transform(single_feats[ifeat]) # misnomer, since the scaling happens later. only here for quick edit            
                full_subj_dict[icond]['single features'][ifeat].append(scaled_X)
            
                if (acc_feat==0) and (acc_mdl==0):

                    full_subj_dict[icond]['Y'] = np.concatenate((full_subj_dict[icond]['Y'], Y_labels[:, np.newaxis]), axis = 0)

            
            full_subj_dict[icond]['single features'][ifeat] = np.concatenate(full_subj_dict[icond]['single features'][ifeat], axis=0)    
                        
            acc_feat += 1
            
        acc_mdl += 1
    
#%%

from scipy.spatial.distance import directed_hausdorff

for icond in expconds:
    
    tmp = full_subj_dict[icond]['single features']
    Yvect = full_subj_dict[icond]['Y'][:, 0]
    
    mat_bothfeats, feat_order = [], []
    for ifeat, matvals in tmp.items():
        
        mat_bothfeats.append(matvals[:, :, np.newaxis])
        feat_order.append(ifeat) 

    mat_bothfeats = np.concatenate(mat_bothfeats, axis=2)
        
    dist_parcels = []
    for iParcel in range(52):
        
        C1 = mat_bothfeats[Yvect==1, iParcel, :]
        C2 = mat_bothfeats[Yvect==2, iParcel, :]
    
        dist_parcels.append(directed_hausdorff(C1, C2)[0])
        
    best_parcel = np.argmax(dist_parcels)
    
    C1_best = mat_bothfeats[Yvect==1, best_parcel, :]
    C2_best = mat_bothfeats[Yvect==2, best_parcel, :]
    
    FullComps = np.concatenate((C1_best, C2_best), axis=0)

    xlim = [-2, 4]    
    if icond == 'ECEO':    
        labels_lit = ['EC']*C1_best.shape[0] + ['EO']*C1_best.shape[0]
        ylim = [-4, 2]
    else:
        labels_lit = ['bsl']*C1_best.shape[0] + ['VS']*C1_best.shape[0]
        ylim = [-2, 4]
        
    tmp_dict = {feat_order[0] : FullComps[:, 0],
                feat_order[1] : FullComps[:, 1],
                'state' : labels_lit}
    
    tmp_DF = pd.DataFrame.from_dict(tmp_dict)
    
    plt.figure()
    sns.kdeplot(data=tmp_DF, x=feat_order[0], y=feat_order[1], hue='state')
    plt.title(icond)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    
    
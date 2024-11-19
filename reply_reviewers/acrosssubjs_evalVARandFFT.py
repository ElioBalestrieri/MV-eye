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
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score

# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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

# various 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/rev_reply_MEG/'

# output folder
outfold = '../STRG_decoding_accuracy/rev_reply_MEG/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

expconds = ['ECEO', 'VS'];
mdltypes = ['TimeFeats', 'FullFFT']
n_cvs = 10 

#%%    

full_dict = {'model' : [], 'condition' : [], 'feature' : [],}  

full_subj_dict = {'subj':[], 'single features':{}, 'Y':[]}
for imdl in mdltypes:

    if imdl == 'TimeFeats':
        featnames = ['MCL', 'std', 'mad', 'DN_FitKernelSmoothraw_entropy', 'iqr']              
    else:
        featnames = ['fullFFT']              
            
    for isubj in range(29):
        
        acc_feat = 0 
        for ifeat in featnames:
        
            if isubj == 0:
                full_subj_dict['single features'].update({ifeat : []})   
                
                if ifeat == 'fullFFT': 
                    full_subj_dict['single features'].update({'totPOW' : []})   
                    
            acc_cond = 0; container_feats = []
            container_totPOW = []
            for icond in expconds:
            
                fname = infold + f'{isubj+1:02d}_{icond}_{imdl}.mat'
    
                mat_content = loadmat_struct(fname)
                F = mat_content['variableName']
                Y_labels = F['Y'] + acc_cond
                single_feats = F['single_feats']
                single_parcels = F['single_parcels']
                        
                tmp_ = single_parcels[0]
                if ifeat == 'fullFFT':        
                    
                    totPOW = tmp_.sum(axis=1)
                    container_totPOW.append(totPOW)
                    feat_red = tmp_

                else:
                    
                    full_featnames = np.array(list(single_feats.keys()))
                    idx_feat = int(np.where(full_featnames==ifeat)[0][0])
                    feat_red = tmp_[:, idx_feat]
                       
                container_feats.append(feat_red)
                
                if acc_feat==0:
                    full_subj_dict['Y'].append(Y_labels[:, np.newaxis])
              
                acc_cond += 2

            if ifeat == 'fullFFT':   
        
                fullFFT = np.concatenate(container_feats, axis=0)
                sclr_mdl = RobustScaler()
                fullFFTscaled = sclr_mdl.fit_transform(fullFFT)
                full_subj_dict['single features'][ifeat].append(fullFFTscaled)

                tmp_CAT = np.concatenate(container_totPOW, axis=0)
                sclr_mdl = RobustScaler()
                tmp_SCALE = sclr_mdl.fit_transform(tmp_CAT[:, np.newaxis])
                full_subj_dict['single features']['totPOW'].append(tmp_SCALE)
            
            else:

                tmp_CAT = np.concatenate(container_feats, axis=0)
                sclr_mdl = RobustScaler()
                tmp_SCALE = sclr_mdl.fit_transform(tmp_CAT[:, np.newaxis])
                full_subj_dict['single features'][ifeat].append(tmp_SCALE)
                
                
                    
            acc_feat+=1
        
        print(f'{icond} {imdl} {ifeat}')
    
                
#%%

tmp_dict = {}
for key, value in full_subj_dict['single features'].items():
    
    if key == 'fullFFT':        
        norm_FFT = np.concatenate(full_subj_dict['single features']['fullFFT'], axis=0)
        
    else:
        tmp_dict.update({key : np.concatenate(value, axis=0)[:, 0]})

DF = pd.DataFrame.from_dict(tmp_dict)

TimeFeatsOI = ['MCL', 'std', 'mad', 'DN_FitKernelSmoothraw_entropy', 'iqr']              

nfreqs = norm_FFT.shape[1]
acc_plot = 0

px = 1/plt.rcParams['figure.dpi'] 
plt.figure(figsize=(800*px,900*px))
for ifeat in TimeFeatsOI:
    
    acc_plot += 1
    plt.subplot(len(TimeFeatsOI), 2, acc_plot)
    sns.scatterplot(data=DF, x='totPOW', y=ifeat, size=.1, alpha=.2)

    mdl_reg = LinearRegression()
    mdl_reg.fit(np.array(DF[ifeat])[:, np.newaxis], np.array(DF['totPOW'])[:, np.newaxis])
    y_pred = mdl_reg.predict(np.array(DF[ifeat])[:, np.newaxis])
    R2out = r2_score(np.array(DF['totPOW'])[:, np.newaxis], y_pred)
    plt.title('R2 : {:0.4}'.format(R2out))

    plt.legend([],[], frameon=False)

    R2_list = []
    for ifreq in range(nfreqs):
        
        mdl_reg = LinearRegression()
        vectTS = np.array(DF[ifeat])[:, np.newaxis]
        vectFFT = norm_FFT[:, ifreq][:, np.newaxis]
        
        mdl_reg.fit(vectTS, vectFFT)
        y_pred = mdl_reg.predict(vectTS)
        
        R2_list.append(r2_score(vectFFT, y_pred))
        
    R2_vect = np.array(R2_list)
    acc_plot +=1
    plt.subplot(len(TimeFeatsOI), 2, acc_plot)
    plt.plot(np.arange(1, 101), R2_vect)
    plt.ylabel('R2')
    if acc_plot==10:
        plt.xlabel('Freq (Hz)')

plt.tight_layout()


#%%

plt.figure()
acc_plot = 0
for ifeat in DF.columns:
    
    acc_plot+=1
    plt.subplot(2, 3, acc_plot)
    sns.histplot(data=DF, x=ifeat)


#%%

# plt.figure()
# plt.subplot(121)
# 
# plt.subplot(122)
# sns.histplot(data=DF, x='std')

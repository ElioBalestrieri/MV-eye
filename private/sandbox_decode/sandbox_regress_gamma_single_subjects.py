# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, zscore

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs, loadmat_struct

#%% folder(s) definition & cond type

#input folder 
infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/Mdl_comparison/'

# output folder
outfold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# current condition type
# this is provided by input function from the parent bash script

ThisExpCond = 'VS'
allCondNames = ['baseline', 'VS']
mdltypes = ['TimeFeats', 'FullFFT'] # 'FreqBandsSimple', 'FreqBands', 'FullFFT', 'TimeFeats'

#%% Pipeline object definition


from sklearn.decomposition import PCA

# regression
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

# crossvalidation
from sklearn.model_selection import cross_val_score

# scaler
from sklearn.preprocessing import RobustScaler 

from sklearn.preprocessing import FunctionTransformer


# pipeline definer
from sklearn.pipeline import Pipeline


# # define the pipeline to be used 
pipe_PLS = Pipeline([('scaler', RobustScaler()),
                     ('squeezer', FunctionTransformer(np.tanh)),
                     ('PLS', PLSRegression(n_components=1))])

pipe_LR = Pipeline([('scaler', RobustScaler()),
                    ('squeezer', FunctionTransformer(np.tanh)),
                     ('LR', LinearRegression())])


# class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
#                                                             strategy='mean')),
#                            ('scaler', RobustScaler()),
#                            ('squeezer', FunctionTransformer(np.tanh)),
#                            ('std_PCA', PCA(n_components=.9, svd_solver='full')),
#                            ('SVM', SVC(C=10))
#                            ])


# # pipeline definer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import balanced_accuracy_score

#%% load files for one participant

collect_datasets = {}
        
for imdl in mdltypes:
    
    data_type = ThisExpCond + '_' + imdl
    filename = '01_' + data_type + '.mat'
    dat = loadmat_struct(infold + filename)

    collect_datasets.update({imdl : dat['variableName']})


#%% parse pre and poststim

ordered_idxs = {}

for imdl in mdltypes:

    lbl_cond = 0
    for icnd in allCondNames:
        
        lbl_cond += 1
        mask_cond = collect_datasets[imdl]['Y']==lbl_cond
        ntrls = int(len(collect_datasets[imdl]['Y'])/2)
    
        unique_trl_id = []
        for itrl in range(ntrls):
            
            mask_trl = collect_datasets[imdl]['trl_order']==(itrl+1)
            this_mask = mask_trl & mask_cond
            unique_trl_id.append(np.where(this_mask)[0][0])

        # append in orderly labeled dict
        ordered_idxs.update({imdl + '_' + icnd : unique_trl_id})

#%% define dep var
# gamma_ranges = np.array([7.9, 13.1])
gamma_ranges = np.array([51, 65])



depvar_full = collect_datasets['FullFFT']
nfreqs = len(depvar_full['FFTpars']['freqs'])
nparcels = len(depvar_full['single_parcels'])

visual_resp = depvar_full['single_feats']['fullFFT'][ordered_idxs['FullFFT_VS'], :]
visual_bsl = depvar_full['single_feats']['fullFFT'][ordered_idxs['FullFFT_baseline'], :]

resp_bslnd = np.log10(visual_resp / visual_bsl)

rshpd_3d = resp_bslnd.reshape((ntrls, nfreqs, nparcels))

gamma_mask = (depvar_full['FFTpars']['freqs']>gamma_ranges.min()) & (depvar_full['FFTpars']['freqs']<gamma_ranges.max())
gamma_resp = rshpd_3d[:, gamma_mask, :].mean(axis=1)

w_p = gamma_resp.mean(axis=0);
PCA_mdl_trgt = PCA(n_components=1)
Y_trgt = PCA_mdl_trgt.fit_transform(gamma_resp)[:, 0] # gamma_resp @ w_p
            
#%% perform regression

R2_scores_PLS, R2_scores_LinReg, correl_coeff = {}, {}, {}
for key in collect_datasets['TimeFeats']['single_feats']:
                
    X_full = collect_datasets['TimeFeats']['single_feats'][key]
    X_reg = X_full[ordered_idxs['TimeFeats_baseline']]

    R2_PLS = cross_val_score(pipe_PLS, X_reg, y=Y_trgt, scoring='r2', cv=5)
    R2_scores_PLS.update({key : R2_PLS.mean()})
    
    # R2_LR = cross_val_score(pipe_LR, X_reg, y=Y_trgt, scoring='r2', cv=5)
    # R2_scores_LinReg.update({key : R2_LR.mean()})

    # "analogic" method: PCA & correlation
    PCA_mdl = PCA(n_components=1)
    tr_X = PCA_mdl.fit_transform(X_reg)[:,0]

    tmp = pearsonr(tr_X, Y_trgt)    
    correl_coeff.update({key : [tmp.statistic, tmp.pvalue]})


#                 regressors[key] = regressors[key][mask_independent, :]
            


# #%% load dependent (fullFFT) and independent (TimeFeats) separately

# for mdltype in mdltypes:
                                
#     fname_X = infold + ThisExpCond + '_allsubjs_X_'  + mdltype + '.pickle'
#     fname_Y = infold + ThisExpCond + '_allsubjs_Y_'  + mdltype + '.pickle'
#     fname_ID = infold + ThisExpCond + '_allsubjs_ID_'  + mdltype + '.pickle'
        
#     # load pickles here
#     with open(fname_X, 'rb') as handle:
#         allsubjs_X = pickle.load(handle)
#     with open(fname_Y, 'rb') as handle:
#         allsubjs_Y = pickle.load(handle)
#     with open(fname_ID, 'rb') as handle:
#         allsubjs_ID = pickle.load(handle)

#     match mdltype:
        
#         case 'FullFFT':

#             # define the experimental condition where the dependnet measure is taken:
#             # visual stimulation
#             mask_dependent = allsubjs_Y == 2

#             # load one dataset for specs
#             # load file & extraction
#             data_type = ThisExpCond + '_' + mdltype
            
#             fname = infold + f'{0+1:02d}_' + data_type + '.mat'
#             mat_content = loadmat_struct(fname)
#             F = mat_content['variableName']
            
#             nfreqs = len(F['FFTpars']['freqs'])
#             nparcels = len(F['single_parcels'])
#             ntrials_full = sum(mask_dependent)

#             visual_resp = allsubjs_X['fullFFT'][mask_dependent, :]
#             rshpd_3d = visual_resp.reshape((ntrials_full, nfreqs, nparcels))
            
#             gamma_mask = (F['FFTpars']['freqs']>55) & (F['FFTpars']['freqs']<70)
#             gamma_resp = rshpd_3d[:, gamma_mask, :].mean(axis=1)

#             # weighted sum of gamma response based on grand average distribution
#             # of gamma power across parcels
#             # w_p = gamma_resp.mean(axis=0);
#             # gamma_resp = gamma_resp @ w_p
            
#             gamma_resp = gamma_resp[:, 0]
            
#         case 'TimeFeats':
            
#             mask_independent = allsubjs_Y == 1
#             regressors = allsubjs_X.copy()
#             del regressors['aggregate']
            
#             for key in regressors:
                
#                 regressors[key] = regressors[key][mask_independent, :]
            

# #%% match prestim with poststim 

# trls_ID_pre = allsubjs_trl_order[mask_independent]-1 # adjust for 0 start in python
# trls_ID_post = allsubjs_trl_order[mask_dependent]-1
# subj_ID_pre = np.array(allsubjs_ID)[mask_independent]
# subj_ID_post = np.array(allsubjs_ID)[mask_dependent]

# unique_trl_id_pre, unique_trl_id_post = [], []

# for subjID in np.unique(subj_ID_pre):
    
#     subjmask_pre = subj_ID_pre == subjID
#     subjmask_post = subj_ID_post == subjID
    
#     ntrls = subjmask_pre.sum()

#     for itrl in range(ntrls):
        
#         this_mask_pre = subjmask_pre & (trls_ID_pre==itrl)
#         this_mask_pos = subjmask_post & (trls_ID_post==itrl)

#         unique_trl_id_pre.append(np.where(this_mask_pre)[0][0])
#         unique_trl_id_post.append(np.where(this_mask_pos)[0][0])


# #%% perform regression

# regr_mdl = LinearRegression()
# ordrd_Y = gamma_resp[unique_trl_id_post]

# R2_scores_regressors = {}
# for key in regressors:
    
#     ordrd_X = regressors[key][unique_trl_id_pre, :]
#     R2 = cross_val_score(regr_mdl, ordrd_X, y=ordrd_Y, scoring='r2', cv=3)
#     R2_scores_regressors.update({key : R2.mean()})



# foo = 1


# #%%


# testX = regressors['SP_Summaries_welch_rect_area_5_1'][:, 0]

# plt.figure()
# plt.scatter(testX, ordrd_Y, s=10)





    
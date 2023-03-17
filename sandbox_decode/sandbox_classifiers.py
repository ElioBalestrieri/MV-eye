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
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)



#%%

# 1-subj sequence

def loop_classify_feats(Xs, Y, pipe, cv_fold=10):
    
    storing_vect = np.empty((1, len(Xs)+1))
    
    count_exceptions = 0
    acc_feat = 0; 
    for key in Xs:
        
        X = Xs[key]
        
        # get rid of full nan columns, if any
        X = X[:, ~(np.any(np.isnan(X), axis=0))]

        # start concatenate arrays for whole freqband decode
        if acc_feat==0:
            catX = np.copy(X)
        else:
            catX = np.concatenate((catX, X), axis=1)
        
        # sometimes there might be a convergence error. avoid breaking the whole
        # computation for this 
        try:        
            acc = cross_val_score(pipe, X, Y, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
        except:
            acc = np.nan; count_exceptions += 1
                
        storing_vect[0, acc_feat] = acc
            
        acc_feat += 1    
        # print(str(acc_feat) + '/' + str(len(Xs)+1) + ' features computed\n')

    # compute accuracy on the whole freqbands, aggregated features. Always with
    # a try statement for the same reason listed above
    try:
        acc_whole = cross_val_score(pipe, catX, Y, cv=cv_fold, 
                                    scoring='balanced_accuracy').mean()
    except:
        acc_whole = np.nan; count_exceptions += 1

    storing_vect[0, acc_feat] = acc_whole
    
    return storing_vect, count_exceptions



def loop_classify_parcels(single_parcels, Y, pipe, cv_fold=10):
    
    acc_parcel = 0;     count_exceptions = 0
    storing_vect = np.empty(len(single_parcels))
    for X in single_parcels:
        
        # get rid of full nan columns, if any
        X = X[:, ~(np.any(np.isnan(X), axis=0))]

        # sometimes there might be a convergence error. avoid breaking the whole
        # computation for this 
        try:        
            acc = cross_val_score(pipe, X, Y, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
        except:
            acc = np.nan; count_exceptions += 1
                
        storing_vect[acc_parcel] = acc

        acc_parcel += 1
        
    return storing_vect, count_exceptions





def single_subj_classify(isubj, infold, outfold):

    # load parcellation labels and select "visual", compatibly with the parcels previously selected i nMATLAB
    HCP_parcels = pd.read_csv('../helper_functions/HCP-MMP1_UniqueRegionList_RL.csv')
    red_HCP = HCP_parcels[HCP_parcels['cortex'].str.contains('visual', case=False)]

    ftypes = ['VG']
    acc_type = 0
    for itype in ftypes:
        
        fname = infold + f'{isubj+1:02d}_' + itype + '_feats.mat'    
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']    
        Y_labels = F['Y']
        single_feats = F['single_feats']
        single_parcels = F['single_parcels']

        # call loop across all features + aggregated feature
        feats_accs_type, count_exc1 = loop_classify_feats(single_feats, Y_labels, 
                                                         class_pipeline)

        parcels_accs_type, count_exc2 = loop_classify_parcels(single_parcels, Y_labels, 
                                                         class_pipeline)
        
        red_HCP['decode_accuracy_' + itype] = parcels_accs_type

        if acc_type ==0:
            storing_mat = np.empty((len(ftypes), len(single_feats)+1))

        storing_mat[acc_type, :] = feats_accs_type
        
        acc_type += 1
    
    updtd_col_list = list(single_feats.keys()); updtd_col_list.append('full_set')
    
    subjDF_feats = pd.DataFrame(storing_mat, columns=updtd_col_list, index=ftypes)
    
    foutname_feats = outfold + f'{isubj+1:02d}_VG_accs_feats.csv' 
    foutname_parcels = outfold + f'{isubj+1:02d}_VG_accs_parcels.csv' 

    subjDF_feats.to_csv(foutname_feats)
    red_HCP.to_csv(foutname_parcels)


#%% call the subject function

for isubj in range(29):

    single_subj_classify(isubj, infold, outfold)
    print(isubj)



# #%% loop pipeline across subjects and features

# # load only 22 subjects out of 29. The remaining 7 will be used as test at a later point in time
# nsubjs_loaded = 22

# for isubj in range(nsubjs_loaded):

#     # load file & extraction
#     fname = f'{isubj+1:02d}' + '_feats.mat'
#     mat_content = loadmat_struct(fname)
#     F = mat_content['variableName']
    
#     Y_labels = F['Y']
    
#     if isubj == 0:        
#         summ_dict = {}
    
#     single_feats = F['single_feats']
    
#     for key in single_feats:
        
#         feat_array = single_feats[key]
        
#         # get rid of full nan columns, if any
#         feat_array = feat_array[:, ~(np.any(np.isnan(feat_array), axis=0))]
        
#         acc = cross_val_score(class_pipeline, feat_array, Y_labels, cv=10, 
#                               scoring='balanced_accuracy').mean()
        
#         if isubj ==0:
            
#             summ_dict.update({key : [np.round(acc, 2)]})

#         else:
            
#             summ_dict[key].append(np.round(acc, 2))
            
#     print('Done wih subj ' + f'{isubj+1:02d}')

# #%% plot stuff

# DF_accs = pd.DataFrame.from_dict(summ_dict)

# # sort DF columns to mean 
# new_idx = DF_accs.mean().sort_values(ascending=False);
# DF_accs = DF_accs.reindex(new_idx.index, axis=1)

# #%% save (or regret that)

# DF_accs.to_csv(path_or_buf='decode_accs.csv')

# #%%

# plt.figure()
# # plt.subplot(121)
# sb.barplot(data=DF_accs, orient='h', palette="ch:start=.2,rot=-.3, dark=.4")
# plt.tight_layout()
# plt.title('Decoding accuracy, ' + str(nsubjs_loaded) + ' participants')
# plt.xlabel('balanced accuracy')

# # subselect the n accs > 90 % and plot only them as violinplots
# feats_accs = DF_accs.mean(); high_perf_feats = feats_accs[feats_accs>.9]
# red_best_DF = DF_accs[high_perf_feats.index]

# plt.figure()
# # plt.subplot(122)
# sb.stripplot(data=red_best_DF, orient='h', palette="ch:start=.2,rot=-.3,dark=.4, light=.8", alpha=.5)
# sb.barplot(data=red_best_DF, orient='h', errcolor=(.3, .3, .3, 1),
#     linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))
# plt.tight_layout()
# plt.title('Best features, ' + str(nsubjs_loaded) + ' participants')
# plt.xlabel('balanced accuracy')
# plt.xlim((.6, 1))

# # plt.xticks(range(len(srtd_accs)), list(srtd_accs.keys()), 
# #            rotation=55, ha='right');

# # #%% sort in ascending order

# # srtd_accs = sorted(summ_dict.items(), key=lambda x:x[1])
# # srtd_accs = dict(srtd_accs)


# # #%% visualize

# # plt.figure();
# # plt.bar(range(len(srtd_accs)), list(srtd_accs.values()), align='center');
# # plt.xticks(range(len(srtd_accs)), list(srtd_accs.keys()), 
# #            rotation=55, ha='right');
# # plt.ylim((.5, 1))
# # plt.tight_layout()
# # plt.ylabel('Accuracy')


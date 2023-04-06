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
import dask
from datetime import datetime

## comment for speed reduction
print(sys.argv)
icond = sys.argv[1]

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

#%% Pipeline object definition

# functions for gaussian mixture modeling
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# pipeline definer
from sklearn.pipeline import Pipeline

# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer

# crossvalidation
from sklearn.model_selection import cross_val_score

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler # to be used at some point?
from sklearn import metrics


# define the grid for gaussian mixture models
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

global param_grid
param_grid = {"n_components": range(1, 7),
              "covariance_type": ["spherical", "tied", "diag", "full"],
              }

# define the pipeline to be used
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


#%% functions

def loop_classify_feats(Xs, Y, pipe, cv_fold=10):
    
    depvars_types = ['decode_acc', 'clust_acc', 'clusts_N']
    storing_mat = np.empty((len(depvars_types), len(Xs)+1))

    # call the single elements of the pipeline
    # to be used outside of the pipeline for clustering, since the clustering
    # does not require crossval 
    impute_missing = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = RobustScaler()  
    grid_search_gauss = GridSearchCV(GaussianMixture(), param_grid=param_grid, 
                              scoring=gmm_bic_score)



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
        
        # try decoding (supervised learning)
        try:        
            acc = cross_val_score(pipe, X, Y, cv=cv_fold, 
                                  scoring='balanced_accuracy').mean()
        except:
            acc = np.nan; count_exceptions += 1
            
        # UPDATE MATRIX
        storing_mat[0, acc_feat] = acc 
             
        # try:
        #
        #     X_cleaned = impute_missing.fit_transform(X)
        #     X_scaled = scaler.fit_transform(X_cleaned)
        #     X_squeezed = np.tanh(X_scaled)
        #
        #     grid_search_gauss.fit(X_squeezed)
        #
        #     N_comps = grid_search_gauss.best_estimator_.n_components;
        #
        #     # evaluate congruence
        #     predicted_labels = grid_search_gauss.best_estimator_.predict(X_squeezed)
        #     acc_clusts = metrics.rand_score(Y, predicted_labels)
        #
        # except:
            
        acc_clusts = np.nan; predicted_labels = np.nan; count_exceptions += 1
        N_comps = np.nan

        # UPDATE MATRIX
        storing_mat[1, acc_feat] = acc_clusts
        storing_mat[2, acc_feat] = N_comps
        
        
        acc_feat += 1    
 
    # compute accuracy on the whole freqbands, aggregated features. Always with
    # a try statement for the same reason listed above
    try:
        acc_whole = cross_val_score(pipe, catX, Y, cv=cv_fold, 
                                    scoring='balanced_accuracy').mean()
    except:
        acc_whole = np.nan; count_exceptions += 1

    # feat index updated on last iter
    storing_mat[0, acc_feat] = acc_whole


    # compute accuracy on the whole freqbands, aggregated features. Always with
    # a try statement for the same reason listed above
    # try:
    #     X_cleaned_whole = impute_missing.fit_transform(catX)
    #     X_scaled_whole = scaler.fit_transform(X_cleaned_whole)
    #     X_squeezed_whole = np.tanh(X_scaled_whole)
    #
    #     grid_search_gauss.fit(X_squeezed_whole)
    #
    #     N_comps_whole = grid_search_gauss.best_estimator_.n_components;
    #
    #     # evaluate congruence
    #     predicted_labels_whole = grid_search_gauss.best_estimator_.predict(X_squeezed_whole)
    #     acc_clusts_whole = metrics.rand_score(Y, predicted_labels_whole)
    #
    # except:
    acc_clusts_whole = np.nan; count_exceptions += 1
    N_comps_whole = np.nan

    storing_mat[1, acc_feat] = acc_clusts_whole
    storing_mat[2, acc_feat] = N_comps_whole

    updtd_col_list = list(Xs.keys()); updtd_col_list.append('full_set')
    
    subjDF_feats = pd.DataFrame(storing_mat, columns=updtd_col_list, index=depvars_types)


    return subjDF_feats, count_exceptions



def loop_classify_parcels(single_parcels, Y, pipe, cv_fold=10):

    impute_missing = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = RobustScaler()  
    grid_search_gauss = GridSearchCV(GaussianMixture(), param_grid=param_grid, 
                              scoring=gmm_bic_score)

    acc_parcel = 0;     count_exceptions = 0
    
    N_optimal_clusts = np.empty(len(single_parcels))
    accuracy_clusts = np.empty(len(single_parcels))
    accuracy_svm = np.empty(len(single_parcels))


    for X in single_parcels:
        
        # get rid of full nan columns, if any
        X = X[:, ~(np.any(np.isnan(X), axis=0))]
            
        # try:
            
        #     X_cleaned = impute_missing.fit_transform(X)
        #     X_scaled = scaler.fit_transform(X_cleaned)
        #     X_squeezed = np.tanh(X_scaled)
    
        #     grid_search_gauss.fit(X_squeezed)

        #     N_comps = grid_search_gauss.best_estimator_.n_components;
            
        #     # evaluate congruence 
        #     predicted_labels = grid_search_gauss.best_estimator_.predict(X_squeezed)
        #     acc_clusts = metrics.rand_score(Y, predicted_labels)

        # except:
            
        acc_clusts = np.nan; N_comps = np.nan; count_exceptions += 1
                      
        # try:        
        #     acc = cross_val_score(pipe, X, Y, cv=cv_fold, 
        #                           scoring='balanced_accuracy').mean()
        # except:
        acc = np.nan; count_exceptions += 1
               
        accuracy_svm[acc_parcel] = acc
        N_optimal_clusts[acc_parcel] = N_comps
        accuracy_clusts[acc_parcel] = acc_clusts

        acc_parcel += 1
        
    return accuracy_svm, accuracy_clusts, N_optimal_clusts, count_exceptions


@dask.delayed
def single_subj_classify(isubj, infold, outfold, icond):

    # load parcellation labels and select "visual", compatibly with the parcels previously selected i nMATLAB
    HCP_parcels = pd.read_csv('../helper_functions/HCP-MMP1_UniqueRegionList_RL.csv')

    # expconds = ['ECEO', 'VS'];
    mdltypes = ['FTM', 'FreqBands', 'FullFFT', 'TimeFeats']
    
    acc_type = 0
    full_count_exc = 0
    # for icond in expconds:
        
    for imdl in mdltypes:

        red_HCP = HCP_parcels[HCP_parcels['cortex'].str.contains('visual', case=False)]

        fname = infold + f'{isubj+1:02d}_' + icond + '_' + imdl + '.mat'
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
        Y_labels = F['Y']
        single_feats = F['single_feats']
        single_parcels = F['single_parcels']

        # call loop across all features + aggregated feature
        subjDF_feats, count_exc1 = loop_classify_feats(single_feats, Y_labels,
                                                         class_pipeline)

        accuracy_svm, parcels_accs_clusts, N_optimal_clusts, count_exc2 = loop_classify_parcels(single_parcels, Y_labels, class_pipeline)

        red_HCP['clustering_accuracy'] = parcels_accs_clusts
        red_HCP['N_clusters_detected'] = N_optimal_clusts
        red_HCP['decoding_accuracy'] = accuracy_svm

        acc_type += 1

        foutname_feats = outfold + f'{isubj+1:02d}_' + icond + '_' + imdl + '_feats_reduced.csv'
        foutname_parcels = outfold + f'{isubj+1:02d}_' + icond + '_' + imdl + '_parcels_reduced.csv'

        red_HCP.to_csv(foutname_parcels)
        subjDF_feats.to_csv(foutname_feats)

        full_count_exc = full_count_exc + count_exc1 + count_exc2

    return full_count_exc

#%% loop pipeline across subjects and features

nsubjs_loaded = 29

allsubjs_DFs = []
for isubj in range(nsubjs_loaded):

    outDF = single_subj_classify(isubj, infold, outfold, icond)
    allsubjs_DFs.append(outDF)
    
#%% actually launch process
countExc = dask.compute(allsubjs_DFs)

# log and save 
countExcDF = pd.DataFrame(list(countExc))
dateTimeObj = datetime.now()
fname = 'log_' + str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '_' + str(dateTimeObj.hour) + '.csv'
countExcDF.to_csv(fname)

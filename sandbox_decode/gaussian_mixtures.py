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

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

#%% Pipeline object definition

# import the main elements needed
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler # to be used at some point?
from sklearn import metrics


# deinfine the pipeline to be used 
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

global param_grid
param_grid = {"n_components": range(1, 7),
              "covariance_type": ["spherical", "tied", "diag", "full"],
              }

#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% functions

def loop_classify_parcels(single_parcels, Y):

    impute_missing = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = RobustScaler()
    
    grid_search_gauss = GridSearchCV(GaussianMixture(), param_grid=param_grid, 
                              scoring=gmm_bic_score)

    acc_parcel = 0;     count_exceptions = 0
    N_optimal_clusts = np.empty(len(single_parcels))
    accuracy_clusts = np.empty(len(single_parcels))

    for X in single_parcels:
        
        # get rid of full nan columns, if any
        X = X[:, ~(np.any(np.isnan(X), axis=0))]
            
        try:
            
            X_cleaned = impute_missing.fit_transform(X)
            X_scaled = scaler.fit_transform(X_cleaned)
            X_squeezed = np.tanh(X_scaled)
    
            grid_search_gauss.fit(X_squeezed)

            N_comps = grid_search_gauss.best_estimator_.n_components;
            
            # evaluate congruence 
            predicted_labels = grid_search_gauss.best_estimator_.predict(X)
            acc = metrics.rand_score(Y, predicted_labels)

        except:
            acc = np.nan; predicted_labels = np.nan; count_exceptions += 1
            
        N_optimal_clusts[acc_parcel] = N_comps
        accuracy_clusts[acc_parcel] = acc

        acc_parcel += 1
        
    return accuracy_clusts, N_optimal_clusts, count_exceptions



def single_subj_classify(isubj, infold, outfold):

    # load parcellation labels and select "visual", compatibly with the parcels previously selected i nMATLAB
    HCP_parcels = pd.read_csv('../helper_functions/HCP-MMP1_UniqueRegionList_RL.csv')
    red_HCP = HCP_parcels[HCP_parcels['cortex'].str.contains('visual', case=False)]

    ftypes = ['NONwhiten_VG', 'PREwhiten_VG']
    acc_type = 0
    for itype in ftypes:
        
        fname = infold + f'{isubj+1:02d}' + itype + '_feats.mat'    
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']    
        Y_labels = F['Y']
        single_parcels = F['single_parcels']

        parcels_accs_type, N_optimal_clusts, count_exc2 = loop_classify_parcels(single_parcels, Y_labels)
        
        red_HCP['clustering_accuracy_' + itype] = parcels_accs_type
        red_HCP['N_clusters_detected' + itype] = N_optimal_clusts
        
        acc_type += 1
        
    foutname_parcels = outfold + f'{isubj+1:02d}_' + itype + '_GM_clustering_parcels.csv' 

    red_HCP.to_csv(foutname_parcels)


#%% call the subject function

for isubj in range(29):

    single_subj_classify(isubj, infold, outfold)
    print(isubj)

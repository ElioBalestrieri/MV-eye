# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
import pandas as pd
import dask
import os
import sys
 
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
from sklearn.preprocessing import RobustScaler 

# define the pipeline to be used 
class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
                                                            strategy='mean')),
                           ('scaler', RobustScaler()),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10))
                           ])

# define fre bands as file strings
freqbands = ['delta_1_4_Hz', 'theta_4_8_Hz', 'alpha_8_13_Hz', 'beta_13_30_Hz', 
             'low_gamma_30_45_Hz', 'high_gamma_55_100_Hz']

#input folder 
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)


#%% define function for calling pipeline on an F structure
# ... once the dataset has been loaded and converted into dict

def loop_classify_feats(single_feats, Y_labels):
    


#%% define the parallel function to be called by dask

@dask.delayed
def single_subj_classify(isubj, infold, outfold):
    
    acc_freq = 0
    for ifreq in freqbands:
        
        # load file & extraction
        print('#######\n\nComputing ' + ifreq + '\n\n#######')
        fname = infold + f'{isubj+1:02d}_' + ifreq + '.mat'
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
    
        Y_labels = F['Y']
        single_feats = F['single_feats']
    
        if acc_freq ==0:
            storing_mat = np.empty((len(freqbands), len(single_feats)+1))
    
        acc_feat = 0; 
        for key in single_feats:
            
            feat_array = single_feats[key]
            
            # get rid of full nan columns, if any
            feat_array = feat_array[:, ~(np.any(np.isnan(feat_array), axis=0))]
    
            # start concatenate arrays for whole freqband decode
            if acc_feat==0:
                swap_feat = np.copy(feat_array)
            else:
                swap_feat = np.concatenate((swap_feat, feat_array), axis=1)
            
            acc = cross_val_score(class_pipeline, feat_array, Y_labels, cv=10, 
                                  scoring='balanced_accuracy').mean()
            
            storing_mat[acc_freq, acc_feat] = acc
                
            acc_feat += 1    
            print(str(acc_feat) + '/' + str(len(single_feats)+1) + ' features computed\n')
    
        # compute accuracy on the whole freqbands, aggregated features
        acc_whole = cross_val_score(class_pipeline, swap_feat, Y_labels, cv=10, 
                                    scoring='balanced_accuracy').mean()
    
        storing_mat[acc_freq, acc_feat] = acc_whole
        
        print(str(len(single_feats)+1) + '/' + str(len(single_feats)+1) + ' features computed\n')
        acc_freq += 1
    
    updtd_col_list = list(single_feats.keys()); updtd_col_list.append('full_set')
    
    subjDF = pd.DataFrame(storing_mat, columns=updtd_col_list, index=freqbands)
    
    foutname = outfold + f'{isubj+1:02d}_freqBands.csv' 
    subjDF.to_csv(foutname)
    
    print('Done wih subj ' + f'{isubj+1:02d}')
    return subjDF
    

#%% loop pipeline across subjects and features

# load only 22 subjects out of 29. The remaining 7 will be used as test at a later point in time
nsubjs_loaded = 22

allsubjs_DFs = []
for isubj in range(nsubjs_loaded):

    outDF = single_subj_classify(isubj, infold, outfold)
    allsubjs_DFs.append(outDF)
    
#%% actually launch process
    
dask.compute(allsubjs_DFs)

#%% plot stuff

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


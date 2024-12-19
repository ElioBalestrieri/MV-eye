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
from temp_file import loadmat_struct
import seaborn as sb
import pandas as pd
import dask

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
infold = '../computed_features/'


#%% loop pipeline across subjects and features

# load only 22 subjects out of 29. The remaining 7 will be used as test at a later point in time
nsubjs_loaded = 2

for isubj in range(nsubjs_loaded):

    if isubj == 0:        
        summ_dict = {}
        strtfd_dict = {}
    
    for ifreq in freqbands:
        
        # load file & extraction
        print('#######\n\nComputing ' + ifreq + '\n\n#######')
        fname = infold + f'{isubj+1:02d}_' + ifreq + '.mat'
        mat_content = loadmat_struct(fname)
        F = mat_content['variableName']
    
        Y_labels = F['Y']
        single_feats = F['single_feats']

        if isubj ==0:
            strtfd_dict.update({ifreq : {}})

        acc_freq = 0    
        for key in single_feats:
            
            feat_array = single_feats[key]
            
            # get rid of full nan columns, if any
            feat_array = feat_array[:, ~(np.any(np.isnan(feat_array), axis=0))]

            # start concatenate arrays for whole freqband decode
            if acc_freq==0:
                swap_feat = np.copy(feat_array)
            else:
                swap_feat = np.concatenate((swap_feat, feat_array), axis=1)
            
            acc = cross_val_score(class_pipeline, feat_array, Y_labels, cv=10, 
                                  scoring='balanced_accuracy').mean()
        
            if isubj == 0:
                strtfd_dict[ifreq].update({key : [np.round(acc, 3)]})
            else:
                strtfd_dict[ifreq][key].append(np.round(acc, 3))

                
            acc_freq += 1    
            print(str(acc_freq) + '/' + str(len(single_feats)+1) + ' features computed\n')

        # compute accuracy on the whole freqbands, aggregated features
        acc_whole = cross_val_score(class_pipeline, swap_feat, Y_labels, cv=10, 
                                    scoring='balanced_accuracy').mean()
        
        if isubj == 0:
            
            strtfd_dict[ifreq].update({'whole_band' : [np.round(acc_whole, 3)]})
            summ_dict.update({ifreq : [np.round(acc_whole, 2)]})
            
        else:
            
            strtfd_dict[ifreq]['whole_band'].append(np.round(acc_whole, 3))
            summ_dict[ifreq].append(np.round(acc_whole, 3))
            
        print(str(len(single_feats)+1) + '/' + str(len(single_feats)+1) + ' features computed\n')
            
            
    print('Done wih subj ' + f'{isubj+1:02d}')








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


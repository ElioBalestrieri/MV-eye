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

# deinfine the pipeline to be used 
class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
                                                            strategy='mean')),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC())
                           ])

#%% loop pipeline across subjects and features

# load only 22 subjects out of 29. The remaining 7 will be used as test at a later point in time
nsubjs_loaded = 22

for isubj in range(nsubjs_loaded):

    # load file & extraction
    fname = f'{isubj+1:02d}' + '_feats.mat'
    mat_content = loadmat_struct(fname)
    F = mat_content['variableName']
    
    Y_labels = F['Y']
    
    if isubj == 0:        
        summ_dict = {}
    
    single_feats = F['single_feats']
    
    for key in single_feats:
        
        feat_array = single_feats[key]
        
        # get rid of full nan columns, if any
        feat_array = feat_array[:, ~(np.any(np.isnan(feat_array), axis=0))]
        
        acc = cross_val_score(class_pipeline, feat_array, Y_labels, cv=10, 
                              scoring='balanced_accuracy').mean()
        
        if isubj ==0:
            
            summ_dict.update({key : [np.round(acc, 2)]})

        else:
            
            summ_dict[key].append(np.round(acc, 2))
            
    print('Done wih subj ' + f'{isubj+1:02d}')

#%% plot stuff

DF_accs = pd.DataFrame.from_dict(summ_dict)

# sort DF columns to mean 
new_idx = DF_accs.mean().sort_values(ascending=False);
DF_accs = DF_accs.reindex(new_idx.index, axis=1)

#%% save (or regret that)

DF_accs.to_csv(path_or_buf='decode_accs.csv')

#%%

plt.figure()
# plt.subplot(121)
sb.barplot(data=DF_accs, orient='h', palette="ch:start=.2,rot=-.3, dark=.4")
plt.tight_layout()
plt.title('Decoding accuracy, ' + str(nsubjs_loaded) + ' participants')
plt.xlabel('balanced accuracy')

# subselect the n accs > 90 % and plot only them as violinplots
feats_accs = DF_accs.mean(); high_perf_feats = feats_accs[feats_accs>.9]
red_best_DF = DF_accs[high_perf_feats.index]

plt.figure()
# plt.subplot(122)
sb.stripplot(data=red_best_DF, orient='h', palette="ch:start=.2,rot=-.3,dark=.4, light=.8", alpha=.5)
sb.barplot(data=red_best_DF, orient='h', errcolor=(.3, .3, .3, 1),
    linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))
plt.tight_layout()
plt.title('Best features, ' + str(nsubjs_loaded) + ' participants')
plt.xlabel('balanced accuracy')
plt.xlim((.6, 1))

# plt.xticks(range(len(srtd_accs)), list(srtd_accs.keys()), 
#            rotation=55, ha='right');

# #%% sort in ascending order

# srtd_accs = sorted(summ_dict.items(), key=lambda x:x[1])
# srtd_accs = dict(srtd_accs)


# #%% visualize

# plt.figure();
# plt.bar(range(len(srtd_accs)), list(srtd_accs.values()), align='center');
# plt.xticks(range(len(srtd_accs)), list(srtd_accs.keys()), 
#            rotation=55, ha='right');
# plt.ylim((.5, 1))
# plt.tight_layout()
# plt.ylabel('Accuracy')


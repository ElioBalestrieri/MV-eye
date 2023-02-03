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
from temp_file import cat_subjs
import seaborn as sb
import pandas as pd


#%% define input folder and feats to be used

infold = '../computed_features/'

DF_accs = pd.read_csv('decode_accs.csv')
new_idx = DF_accs.mean().sort_values(ascending=False)
DF_accs = DF_accs.reindex(new_idx.index, axis=1)

# clear cols with mean > 1 & spctr_fooof
mask_keep = (DF_accs.mean()<1) & (DF_accs.columns != 'spctr_fooof')
DF_accs = DF_accs.loc[:,mask_keep]

# select first 10 columns
best_feats = DF_accs.columns[0:10]

hyperoptimizeflag = False


#%% generate datasets

full_train_X, full_train_Y = cat_subjs(infold, best_feats, strtsubj=0, endsubj=22)
test_leftout_X, test_leftout_Y = cat_subjs(infold, best_feats, strtsubj=22, endsubj=29)


#%% Pipeline object definition

# import the main elements needed
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score

# pipeline definer
from sklearn.pipeline import Pipeline

# hyperparameters estim
from sklearn.model_selection import ShuffleSplit, KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange

#%% create Hyperpipe

if hyperoptimizeflag:

    pipe = Hyperpipe('optimize_SVC',
                 optimizer='random_grid_search',
                 optimizer_params={'limit_in_minutes': 5},
                 outer_cv=ShuffleSplit(test_size=0.2, n_splits=3),
                 inner_cv=KFold(n_splits=5, shuffle=True),
                 metrics=['accuracy', 'precision', 'recall', "f1_score"],
                 use_test_set=True,
                 verbosity=0)


    pipe += PipelineElement('PCA', hyperparameters={
                            'n_components': FloatRange(0.5, 0.9, step=0.1)})

    pipe += PipelineElement('SVC', hyperparameters={'C': [.01, .1, 1, 10, 100]})

    pipe.fit( full_train_X['mean'],  full_train_Y)


#%% define the pipeline to be used 
# C = 10 comes from a qick & dirty hyperparameter optimization, 
# but more work will be needed in this direction

class_pipeline = Pipeline([('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(verbose=True, C=10))
                           ])

#%% loop the 10 + 1 best features for crossvalidated and leavout accuracy

DF_fulldata_SVM = pd.DataFrame(columns=full_train_X.keys())


for ifeat in full_train_X:
    
    print('\n####################################################')
    print('\n\ncompute ' + ifeat)
    print('\n\n')
    
    acc_train_CV = cross_val_score(class_pipeline, full_train_X[ifeat], full_train_Y, cv=10, 
                                   scoring='balanced_accuracy').mean()
    
    # store CV
    DF_fulldata_SVM.loc[0, ifeat] = acc_train_CV
    
    # test and store leftout
    pipe_mdl_fit = class_pipeline.fit(full_train_X[ifeat], full_train_Y)
    leftout_preds_Y = pipe_mdl_fit.predict(test_leftout_X[ifeat])
    leftout_acc = balanced_accuracy_score(test_leftout_Y, leftout_preds_Y)
    DF_fulldata_SVM.loc[1, ifeat] = leftout_acc
    



#%% save (or regret that)

DF_fulldata_SVM.to_csv(path_or_buf='SVM_cross_out_accs.csv')



#%% plot

# sort in descending order
new_idx = DF_fulldata_SVM.mean().sort_values(ascending=False)
DF_fulldata_SVM = DF_fulldata_SVM.reindex(new_idx.index, axis=1)

# allow stacked long version for nicer barplotting
new_index = ['crossval_acc', 'leavout_acc']
DF_fulldata_SVM['acc_type'] = new_index
long_DF = DF_fulldata_SVM.set_index('acc_type').stack().to_frame().reset_index()\
           .rename(columns={'level_1': 'feature', 0: 'accuracy'})

long_DF['accuracy'] = long_DF['accuracy'].astype(float) # to allow rounding
long_DF = long_DF.round(3)


plt.figure()
ax = sb.barplot(data=long_DF.round(3), y='feature', x='accuracy', hue='acc_type', orient='h',  
                palette="ch:start=.2,rot=-.3, dark=.4", ci=None)
plt.tight_layout()
plt.title('SVM Decoding accuracy, pooled participants')
plt.xlabel('balanced accuracy')
plt.xlim((.5, .9))

for container in ax.containers:
    ax.bar_label(container)


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


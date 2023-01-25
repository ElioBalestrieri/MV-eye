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

#%% load file & extraction

fname = 'test.mat'
mat_content = loadmat_struct(fname)
F = mat_content['F']



#%% Pipeline object definition

# import the main elements needed
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# crossvalidation
from sklearn.model_selection import cross_val_score

# pipeline definer
from sklearn.pipeline import Pipeline


# deinfine the pipeline to be used 
class_pipeline = Pipeline([('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC())
                           ])

#%% loop pipeline across features
Y_labels = F['Y']
summ_dict = {}

single_feats = F['single_feats']

for key in single_feats:
    
    feat_array = single_feats[key]
    acc = cross_val_score(class_pipeline, feat_array, Y_labels, cv=10, 
                          scoring='balanced_accuracy').mean()
    
    summ_dict.update({key : np.round(acc, 2)})

#%% sort in ascendin gorder

srtd_accs = sorted(summ_dict.items(), key=lambda x:x[1])
srtd_accs = dict(srtd_accs)


#%% visualize

plt.figure();
plt.bar(range(len(srtd_accs)), list(srtd_accs.values()), align='center');
plt.xticks(range(len(srtd_accs)), list(srtd_accs.keys()), 
           rotation=55, ha='right');
plt.ylim((.5, 1))
plt.tight_layout()
plt.ylabel('Accuracy')


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


#%% load file & extraction

fname = 'test.mat'
mat_content = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
F = mat_content['F']

#%% decoding variable definition

SAMPEN = F.single_feats.SAMPEN
spctr_fooof = F.single_feats.spctr_fooof
covFFT = F.single_feats.covFFT
Y_labels = F.Y

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

print(cross_val_score(class_pipeline, SAMPEN, Y_labels, cv=10, 
                      scoring='balanced_accuracy').mean())

print(cross_val_score(class_pipeline, spctr_fooof, Y_labels, cv=10, 
                      scoring='balanced_accuracy').mean())

print(cross_val_score(class_pipeline, covFFT, Y_labels, cv=10, 
                      scoring='balanced_accuracy').mean())

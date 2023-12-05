#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:39:13 2023

@author: balestrieri
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# parent folder
path = '../STRG_decoding_accuracy/alphaSPADE/timeresolved/'
subjIDs = os.listdir(path)

#%% loop across subjs

acc_subj = 0
for ID in subjIDs:

    this_path = path + ID + '/'

    FilesList = os.listdir(this_path)

    if acc_subj==0:        
        DecAcc_merged = {}

    for iFile in FilesList:
        
        # initializing substrings
        sub1 = ID + '_'
        sub2 = '.csv'
         
        # getting index of substrings
        idx2 = iFile.index(sub2)
        
        # get current feature name, based on filename
        featName = iFile[len(sub1): idx2]
 
        # load file
        tempDF = pd.read_csv(this_path + iFile)
        ntps = tempDF.shape[0]

        if acc_subj==0:                    
            DecAcc_merged[featName] = np.empty((ntps, len(subjIDs)))
            
        DecAcc_merged[featName][:, acc_subj] = tempDF['delta_accuracy']

    if acc_subj==0:
        
        xTime = tempDF['time_winCENter']
        
    acc_subj+=1
    print(ID)


#%% plot accuracy
AVG_accs_dict = {}
for key, array in DecAcc_merged.items():
    
    avg_ = np.mean(array, axis=1)
    stderr = np.std(array, axis=1)/np.sqrt(len(subjIDs))
    
    AVG_accs_dict[key] = avg_
    
    plt.figure()
    
    line_1, = plt.plot(xTime, avg_, 'b-')
    fill_1 = plt.fill_between(xTime, avg_-stderr, avg_+stderr, color='b', alpha=0.2)    
    plt.margins(x=0)

    plt.legend([(line_1, fill_1)], ['delta accuracy from random'])
    plt.title(key)
    
    
#%% compute correlation between features (avg)

DF_avg = pd.DataFrame.from_dict(AVG_accs_dict)
corrfeats = DF_avg.corr()

# Draw the heatmap with the mask and correct aspect ratio
plt.figure()
cmap = sns.diverging_palette(250, 30, l=65, as_cmap=True)
sns.heatmap(corrfeats, square=True, vmax=1, vmin=-1, linewidths=0, cmap=cmap)



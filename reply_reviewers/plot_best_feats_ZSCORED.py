#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:42:52 2023

@author: elio
"""

#%% import and folder definition

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

infold = '../STRG_decoding_accuracy/rev_reply_MEG_zscore/'
plt.close('all')



#%% 3. plot the best feats within subjects and bands

DF = pd.read_csv(infold + 'across_accuracy_FINAL_ZSCORED.csv')


exp_conds = ['ECEO', 'VS']
cols_bind = ['model', 'feature']; 

n_top_feats = 10

for iExpCond in exp_conds:
    
    
    tmp_DF = DF.loc[DF['condition']==iExpCond, :]

    tmp_DF['combined'] = tmp_DF[cols_bind].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1)

    # usual long to wide
    best_feats_wide = pd.pivot(tmp_DF, index='cvFolds', 
                               columns='combined', values='accuracy')    

    new_idx = best_feats_wide.mean().sort_values(ascending=False)[:n_top_feats]
    best_feats_sorted = best_feats_wide.reindex(new_idx.index, axis=1)

    plt.figure()

    ax = sns.stripplot(data=best_feats_sorted, orient='h', 
                       palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
    ax = sns.barplot(data=best_feats_sorted, orient='h', errcolor=(.3, .3, .3, 1),
        linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))
    
    ax.set_yticklabels(best_feats_sorted.columns, size=8)

    plt.title(iExpCond + '\nBest features')
    plt.xlabel('balanced accuracy')
    plt.ylabel('feature')
    plt.xlim((.6, 1))

    cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=len(new_idx))
    acc_ypos = .15; acc_color = 0
    for itxt in new_idx:
        
        plt.text(.6, acc_ypos, str(round(itxt, 3)), color=cpal[acc_color])
        acc_ypos += 1; acc_color += 1

    plt.tight_layout()


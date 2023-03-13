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

infold = '../STRG_decoding_accuracy/'
plt.close('all')

#%% single subjects

up_to_subj = 29

for isubj in range(up_to_subj):
    
    subjname = f'{isubj+1:02d}'
    fname_whit_state = infold + subjname + '_WHIT_vs_NOWHIT_feats.csv'
    DF_whit_state = pd.read_csv(fname_whit_state)

    # long format needed because difference in columns between full freq and bandpassed signal
    long_DF_whit_state = DF_whit_state.set_index('Unnamed: 0').stack().to_frame().\
                    reset_index().rename(columns={'Unnamed: 0': 'whitening_state', 
                                                  'level_1': 'feature', 
                                                  0: 'accuracy'})

    long_DF_whit_state.insert(0, 'SubjID', subjname)
    
    if isubj == 0:
        allsubjs_DF = long_DF_whit_state.copy()
    else:
        allsubjs_DF = pd.concat([allsubjs_DF, long_DF_whit_state])
        
        
allsubjs_DF.to_csv(infold + 'allsubjs_WHIT_vs_NOWHIT.csv')
    

#%% compute summary statistics 

res = pg.rm_anova(data=allsubjs_DF, dv='accuracy', 
                  within=['whitening_state', 'feature'], subject='SubjID',
                  detailed=True)

allsubjs_DF.groupby('whitening_state').describe()
    

fullsets = allsubjs_DF.loc[allsubjs_DF['feature']=='full_set', :]

res_T = pg.ttest(fullsets['accuracy'][fullsets['whitening_state']=='WHIT'], 
                 fullsets['accuracy'][fullsets['whitening_state']=='NOWHIT'],
                 paired=True)

plt.figure()
sns.barplot(data=fullsets, x='whitening_state', y='accuracy', errorbar='se', 
            palette='Set3')
plt.ylim([.9, 1])
plt.title('All Features, whitened vs non-whitened \n' + 
           't=' + str(np.round(res_T['T'][0], decimals=3)) + ', p='  + 
           str(np.round(res_T['p-val'][0], decimals=3)))


#%% plotting
rounded_DF = allsubjs_DF.round(3)

# 2. for each freq band plot the ordered full set of features.
# moreover, select the best 3 features in each freq band
whit_types_names = rounded_DF.whitening_state.unique()

acc_type = 0
for itype in whit_types_names:

    plt.figure()     
    this_DF = rounded_DF.loc[rounded_DF['whitening_state']==itype, :]

    # go to wide format to allow sorting
    wide_DF = pd.pivot(this_DF, index='SubjID', columns='feature', 
                       values='accuracy')    
    new_idx = wide_DF.mean().sort_values(ascending=False)
    wide_DF = wide_DF.reindex(new_idx.index, axis=1)
    
    # plot
    ax = sns.barplot(data=wide_DF, orient='h',
                     palette="ch:start=.2,rot=-.3, dark=.4")
    
    plt.tight_layout()
    plt.title('SVM acc, ' + itype)
    plt.xlabel('balanced accuracy')
    plt.xlim((.5, 1))
    
    # collect either the top 2 features OR all the features > .90 
    # for the current bandpass condition
    winning_feats = new_idx.index[new_idx>.9]
    if winning_feats.empty:
        winning_feats = new_idx.index[0:2]
    
    this_band_best_feats = this_DF[this_DF.feature.isin(winning_feats)]

    cols_bind = ['whitening_state', 'feature']; swap_feats = this_band_best_feats.copy()
    swap_feats['combined'] = this_band_best_feats[cols_bind].apply(
        lambda row: '\n'.join(row.values.astype(str)), axis=1)
    
    if acc_type == 0:
        allsubjs_and_types_best = swap_feats.copy()
    else:
        allsubjs_and_types_best = pd.concat([allsubjs_and_types_best, 
                                             swap_feats])

    acc_type += 1

#%% 3. plot the best feats within subjects and bands

# usual long to wide
best_feats_wide = pd.pivot(allsubjs_and_types_best, index='SubjID', 
                           columns='combined', values='accuracy')    
new_idx = best_feats_wide.mean().sort_values(ascending=False)
best_feats_sorted = best_feats_wide.reindex(new_idx.index, axis=1)

plt.figure()

ax = sns.stripplot(data=best_feats_sorted, orient='h', 
                   palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
ax = sns.barplot(data=best_feats_sorted, orient='h', errcolor=(.3, .3, .3, 1),
    linewidth=1, edgecolor=(.3, .3, .3, 1), facecolor=(0, 0, 0, 0))

ax.set_yticklabels(best_feats_sorted.columns, size=8)


plt.tight_layout()
plt.title('Winning features, (within subjects)')
plt.xlabel('balanced accuracy')
plt.ylabel('freq band / feature')
plt.xlim((.6, 1))

cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=len(new_idx))
acc_ypos = .15; acc_color = 0
for itxt in new_idx:
    
    plt.text(.6, acc_ypos, str(round(itxt, 3)), color=cpal[acc_color])
    acc_ypos += 1; acc_color += 1



plt.show()
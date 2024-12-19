#!/usr/bin/env python
# coding: utf-8

# In[1]:


# basic imports
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


# In[2]: preamble

def summary_class_acc(infold, ntop=12, pivotflag=False):

    filelist = glob.glob(infold+'*.csv')
    
    acc = 0
    for ifile in filelist:
        
        tmp_DF = pd.read_csv(ifile)
        if pivotflag:    
            # tmp_DF = tmp_DF.pivot_table(columns='feature', values='balanced_accuracy')
            tmp_DF = tmp_DF.pivot_table(columns='feature', values='delta_accuracy')
        
        if acc==0:
            DF_tot = tmp_DF

        else:                
            DF_tot = pd.concat([DF_tot, tmp_DF], ignore_index=True, sort=False)
    
        acc+=1
    
    DF_clean = DF_tot.dropna(axis=1)
    
    try:
        DF_clean = DF_clean.drop(['Unnamed: 0'], axis=1) ############################################### return
    except:
        foo = 1
        
    avg_acc = DF_clean.mean().sort_values(ascending=False) ######################################### return
    
    
    topNfeats = avg_acc[0:ntop]
    
    topFeatsNames = list(topNfeats.index)
    nonRedundantTopFeats, tmp_code = [], []
    
    for iFeat in topFeatsNames:
        
        coreID = iFeat[0:6] # assume equality of first 6 elements in string as good proxy for redundancy
        
        if coreID not in tmp_code:
            
            tmp_code.append(coreID)
            nonRedundantTopFeats.append(iFeat)
            
            
    red_DF = DF_clean.loc[:, nonRedundantTopFeats]
    long_red_DF = red_DF.melt(ignore_index=False, value_name = 'delta accuracy', var_name = 'feature') ############# return
    
    stat_summary = {}
    for iFeat in nonRedundantTopFeats:
        
        smpl = red_DF.loc[:, iFeat]
        out = ttest_1samp(smpl, 0, alternative='greater')
        stat_summary[iFeat] = {'stat' : out.statistic,
                               'p' : out.pvalue,
                               'avg' : smpl.mean()}
        
    stats_DF = pd.DataFrame(stat_summary) ################################################### returnable

    out = {'DF_clean' : DF_clean,
           'red_DF' : long_red_DF,
           'full_avg_acc' : avg_acc,
           'red_stats' : stats_DF}
    
    return out

#%% fetch results

sklearn_infold = '../STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim/'    
out_sklearn = summary_class_acc(sklearn_infold, ntop=12, pivotflag=False)

matlab_infold = '../STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim_linearclass/'  
out_matlab = summary_class_acc(matlab_infold, ntop=12, pivotflag=True)

matlab_infold = '../STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim_linearclass_extended/'  
out_matlab = summary_class_acc(matlab_infold, ntop=12, pivotflag=True)




#%% plot sorted sklearn accuracies and corresponding matlab accuracies

srtd_sk = out_sklearn['full_avg_acc']
unsrtd_matlab = out_matlab['full_avg_acc'][srtd_sk.index]

plt.figure()
plt.scatter(np.arange(len(srtd_sk)), np.array(srtd_sk), s=2, label='sklearn')
plt.scatter(np.arange(len(srtd_sk)), np.array(unsrtd_matlab), s=2, label='matlab', alpha=.4)
plt.legend()
plt.xlabel('feature (sklearn-sorted)')
plt.ylabel('delta accuracy')

#%%

plt.figure()
ax = sns.pointplot(data=out_sklearn['red_DF'], x='delta accuracy', y='feature', 
                    errorbar=None, color=".3", linestyle="none", 
                    marker="|", markersize=10)
ax = sns.stripplot(data=out_sklearn['red_DF'], x='delta accuracy', y='feature', 
                    palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
plt.title('Top features (sklearn)')

ntops = out_sklearn['red_stats'].shape[1]
cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=ntops)
acc_ypos = .2; acc_color = 0
for itxt in range(ntops):
    
    acc_txt = 'acc: ' + str(round(out_sklearn['red_stats'].iloc[2, itxt], 3))
    tval_txt = 't val: ' + str(round(out_sklearn['red_stats'].iloc[0, itxt], 3))
    plt.text(-.2, acc_ypos-.2, acc_txt, color=cpal[acc_color])
    plt.text(-.2, acc_ypos, tval_txt, color=cpal[acc_color])
    
    acc_ypos += 1
    acc_color += 1

plt.xlim(left=-.21)
plt.tight_layout()


#%%



plt.figure()
ax = sns.pointplot(data=out_matlab['red_DF'], x='delta accuracy', y='feature', 
                    errorbar=None, color=".3", linestyle="none", 
                    marker="|", markersize=10)
ax = sns.stripplot(data=out_matlab['red_DF'], x='delta accuracy', y='feature', 
                    palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
plt.title('Top features (matlab)')

ntops = out_matlab['red_stats'].shape[1]
cpal = sns.color_palette("ch:dark=.25,light=.5,hue=2",  n_colors=ntops)
acc_ypos = .2; acc_color = 0
for itxt in range(ntops):
    
    acc_txt = 'acc: ' + str(round(out_matlab['red_stats'].iloc[2, itxt], 3))
    tval_txt = 't val: ' + str(round(out_matlab['red_stats'].iloc[0, itxt], 3))
    plt.text(-.3, acc_ypos-.2, acc_txt, color=cpal[acc_color])
    plt.text(-.3, acc_ypos, tval_txt, color=cpal[acc_color])
    
    acc_ypos += 1
    acc_color += 1

plt.xlim(left=-.31)
plt.tight_layout()






#%%

# #%% load data from linearclass (MATLAB)

# # input folder
# infold2 = '../STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim_linearclass/'    
# filelist2 = glob.glob(infold2+'*.csv')

# acc = 0
# for ifile in filelist2:
    
#     tmp_DF = pd.read_csv(ifile)
#     tmp_DF = tmp_DF.pivot_table(columns='feature', values='balanced_accuracy')
    
#     if acc==0:
        
#         DF_tot = tmp_DF
        
#     else:
            
#         DF_tot = pd.concat([DF_tot, tmp_DF], ignore_index=True, sort=False)

#     acc+=1

# #%%

# DF_clean = DF_tot.dropna(axis=1)

# avg_acc = DF_clean.mean().sort_values(ascending=False)

# plt.figure()
# plt.plot(np.array(avg_acc))

# #%%



# ntop = 30
# topNfeats = avg_acc[0:ntop]

# topFeatsNames = list(topNfeats.index)
# nonRedundantTopFeats, tmp_code = [], []

# for iFeat in topFeatsNames:
    
#     coreID = iFeat[0:6] # assume equality of first 6 elements in string as good proxy for redundancy
    
#     if coreID not in tmp_code:
        
#         tmp_code.append(coreID)
#         nonRedundantTopFeats.append(iFeat)
        
        
# red_DF = DF_clean.loc[:, nonRedundantTopFeats]



# long_red_DF = red_DF.melt(ignore_index=False, value_name = 'delta accuracy', var_name = 'feature')

# plt.figure()
# # sns.swarmplot(data=long_red_DF, x='delta accuracy', y='feature')
# ax = sns.pointplot(data=long_red_DF, x='delta accuracy', y='feature', 
#                    errorbar=None, color=".3", linestyle="none", 
#                    marker="|", markersize=10)
# ax = sns.stripplot(data=long_red_DF, x='delta accuracy', y='feature', 
#                    palette="ch:dark=.25,light=.5,hue=2", alpha=.4)
# plt.tight_layout()

# #%%

# stat_summary = {}
# for iFeat in nonRedundantTopFeats:
    
#     smpl = red_DF.loc[:, iFeat]
#     out = ttest_1samp(smpl, 0, alternative='greater')
#     stat_summary[iFeat] = {'stat' : out.statistic,
#                            'p' : out.pvalue}
    
# stats_DF = pd.DataFrame(stat_summary)





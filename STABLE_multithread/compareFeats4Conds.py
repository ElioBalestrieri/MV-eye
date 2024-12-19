#!/usr/bin/env python
# coding: utf-8

# In[1]:


# notebook for 4 conds features benchmarking
import sys
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# from openTSNE import TSNE

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import cat_subjs_train_test, cat_subjs_train_test_ParcelScale

#in/output folders 
infold = '../STRG_computed_features/Mdl_comparison/'
outfold = '../STRG_decoding_accuracy/Mdl_comparison/'

# recompute data merging & classifications? (this step requires time)
reclassify_flag = True

# define function to adapt train and test sets
def adapt_data(dat, deleteFull=True):
    
    # first delete the "full_set" from the copied dictionary    
    # corr_Xtr = fullX_train.copy()

    if deleteFull:
        del dat['full_set']

    # split fooof into offset and slope
    fooof_aperiodic = dat['fooof_aperiodic']
    dat['aperiodic_slope'] = fooof_aperiodic[:, 0::2]
    dat['aperiodic_offset'] = fooof_aperiodic[:, 1::2]
    # and delete fooof aperiodic
    del dat['fooof_aperiodic']

    return dat


# In[2]:


# load & cat files from TimeFeats
list_ExpConds = ['ECEO_TimeFeats', 'VS_TimeFeats']
fullX_train, fullX_test, Y_train_A, Y_test_A, subjID_trials_labels = cat_subjs_train_test(infold, strtsubj=0, endsubj=29, 
                                                                                          ftype=list_ExpConds, tanh_flag=True, 
                                                                                          compress_flag=True)     

# In[ ]:
if reclassify_flag:

    # store previous values, from nonbandpassed signal, and add the features computed in the frequency bands
    temp = {'Xtrain' : adapt_data(fullX_train, deleteFull=False),
            'Xtest' : adapt_data(fullX_test, deleteFull=False),
            'Ytrain' : Y_train_A,
            'Ytest' : Y_test_A,
            'subjIDs' : subjID_trials_labels}
    
    compareBands = {'no_bandpass' : temp}
    
    # loop to load also the bandpassed signals
    freqBands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
    infold_bands = '../STRG_computed_features/TimeFeats_bandpassed/'
    
    for thisBand in freqBands:
        
        this_list_expConds = ['ECEO_' + thisBand + '_TimeFeats', 'VS_' + thisBand + '_TimeFeats']
        Xtr, Xte, Ytr, Yte, subjID_trials_labels = cat_subjs_train_test(infold_bands, strtsubj=0, endsubj=29, 
                                                                        ftype=this_list_expConds, tanh_flag=True, 
                                                                        compress_flag=True)     
        temp = {'Xtrain' : adapt_data(Xtr, deleteFull=False),
                'Xtest' : adapt_data(Xte, deleteFull=False),
                'Ytrain' : Ytr,
                'Ytest' : Yte,
                'subjIDs' : subjID_trials_labels}
    
        temp_dict = {thisBand : temp}
        compareBands.update(temp_dict)
        
        print(thisBand + ' completed')

    # In[ ]:
        
    # load & cat files from freqbands set
    list_ExpConds = ['ECEO_FreqBands', 'VS_FreqBands']
    Xtr, Xte, Ytr, Yte, subjID_trials_labels = cat_subjs_train_test(infold, strtsubj=0, endsubj=29, 
                                                                        ftype=list_ExpConds, tanh_flag=True, 
                                                                        compress_flag=True)     
    temp = {'Xtrain' : Xtr,
            'Xtest' : Xte,
            'Ytrain' : Ytr,
            'Ytest' : Yte,
            'subjIDs' : subjID_trials_labels}
    
    compareBands.update({'freqBands' : temp})
    print('freqBands completed')

    # In[ ]:
     
    # Classifier 1
    # - standardization, across subject, along repetitions.
    
    rbf_svm = SVC(C=10, random_state=42)
    
    blist = []; accfreq = 0
    for bandName, dataset in compareBands.items():
        
        X_train = dataset['Xtrain']
        X_test = dataset['Xtest']
        Y_train = dataset['Ytrain']
        Y_test = dataset['Ytest']
        
        dict_accs = {}
        for key, Xtr in X_train.items():
    
            # fit the SVM model
            this_mdl = rbf_svm.fit(Xtr, Y_train)
    
            # generate predictions & compute balanced accuracy
            Xte = X_test[key]
            pred_labels = this_mdl.predict(Xte)
            this_acc = balanced_accuracy_score(Y_test, pred_labels)
    
            # print some feedback in the CL
            print(bandName + ' ' + key + ': ' + str(round(this_acc, 4)))
    
            # append 
            dict_accs.update({key:this_acc})
    
        DFband = pd.DataFrame(dict_accs, index=[bandName])
        DFband.to_csv(outfold + bandName + '_4conds_collapsed_reproduce_fooofsplit.csv') # save, just to be sure
        blist.append(DFband)
        
    allBandsDF = pd.concat(blist)
    allBandsDF.to_csv(outfold + 'freqBands_TS_4conds_collapsed_reproduce_fooofsplit.csv')


                                                                                        
#%%


# transfor train and test sets
corr_Xtr = copy.deepcopy(fullX_train)
corr_Xte = copy.deepcopy(fullX_test)

print(corr_Xtr.keys())

corr_array, cosine_array = np.zeros((len(corr_Xtr), len(corr_Xtr))), np.zeros((len(corr_Xtr), len(corr_Xtr)))

# start loop to:
# 1: compute classification accuracy of every feature
# 2: compute correlation between features

rbf_svm = SVC(C=10, random_state=42)

acc_row = 0
vect_accuracy = [] # keep is as a list, easier to resort it accoridng to the dendrogram leaves

for rowFeat, FeatArray1 in corr_Xtr.items():
    
    # fit the SVM model
    this_mdl = rbf_svm.fit(FeatArray1, Y_train_A)

    # generate predictions & compute balanced accuracy
    Xte = corr_Xte[rowFeat]
    pred_labels = this_mdl.predict(Xte)
    this_acc = balanced_accuracy_score(Y_test_A, pred_labels)

    vect_accuracy.append(this_acc)
    
    # linearize matrix, to correlate across trials & parcels
    rowFlat = np.hstack(FeatArray1)
    
    acc_col = 0
    for colFeat, FeatArray2 in corr_Xtr.items():
    
        colFlat = np.hstack(FeatArray2)

        corr_array[acc_row, acc_col] = np.corrcoef(rowFlat, colFlat)[0, 1]
        
        acc_col += 1
        
    acc_row += 1
    print(rowFeat)

# round correlation to 5th element. This avoids asymmetric matrix for values close to machine precision
corr_array = np.round(corr_array, 5)


#%%

# start clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

dissimilarity_mat = 1-np.abs(corr_array)

Z = linkage(squareform(dissimilarity_mat), 'complete')


plt.figure()
R = dendrogram(Z, labels=list(corr_Xtr.keys()), orientation='top', 
            leaf_rotation=90, color_threshold=.75);
plt.tight_layout()

print(R['leaves'])


# In[ ]:


# Clusterize the data
threshold = .8
labels = fcluster(Z, threshold, criterion='distance')

# Show the cluster
print(labels)
list(corr_Xtr.keys())

sorted_accs = [vect_accuracy[i] for i in R['leaves']] 

print(sorted_accs)

#%%

from mpl_toolkits.axes_grid1 import make_axes_locatable

DF_corr = pd.DataFrame(data=1-abs(corr_array), index=corr_Xtr.keys(), columns=corr_Xtr.keys())

# sns.clustermap(DF_corr, method="complete", cmap='RdBu_r', xticklabels=True, yticklabels=True,
#               annot_kws={"size": 7}, vmin=-1, vmax=1);

g = sns.clustermap(DF_corr, method="complete", cmap='Reds_r', xticklabels=True, yticklabels=True,
                    annot_kws={"size": 7}, vmin=0, vmax=1, cbar_kws={'orientation' : 'horizontal'},
                  row_linkage=Z, col_linkage=Z);

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width-.02, 0.02])
g.ax_cbar.set_title('distance \n(1-abs(r))')
g.ax_col_dendrogram.set_visible(False)


# In[ ]:


# how many top features?
nfeats = 20
outfold = '../STRG_decoding_accuracy/Mdl_comparison/'

# fetch data & plot
df_power = pd.read_csv(outfold + 'freqBands_4conds_collapsed_reproduce_fooofsplit.csv', index_col=0)
df_TS = pd.read_csv(outfold + 'freqBands_TS_4conds_collapsed_reproduce_fooofsplit.csv', index_col=0)

# drop last entries of the TS df (redundant with power DF)
df_TS = df_TS.iloc[0:-1, 0:-6]

# make the last entry (all bands together) as the first entry (to match the "no bandpass" row in the full DF)
array = np.roll(np.asarray(df_power), 1)
df_TS[''] = np.array([np.nan]*7)
df_TS['power'] = array.T

full_set = df_TS.pop('full_set')
df_TS['full_set'] = full_set
df_TS = df_TS.rename(index={'no_bandpass':'broadband'})

# Convert to Long Format
melted_df = pd.melt(df_TS, var_name='features', value_name='balanced accuracy', ignore_index=False)

# Sort in Descending Order and Rename 'index' to 'frequency'
sorted_df = melted_df.sort_values(by='balanced accuracy', ascending=False).reset_index().rename(columns={'index': 'frequency'})

# merge columns
sorted_df['freq_feat'] = sorted_df['frequency'] + '\n' + sorted_df['features']

# Select the first 20 entries
top_feats = sorted_df.head(nfeats)

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x='balanced accuracy', y='freq_feat', data=top_feats, orient='h',
            palette="ch:start=.2,rot=-.3, dark=.4")

plt.xlabel('balanced accuracy')
plt.ylabel('Features')
plt.title('Top ' + str(nfeats) + ' features' + '\n4 conditions classification')
plt.show()
plt.xlim([.25, 1])
plt.tight_layout()
plt.legend(loc='lower right')


print(top_feats)


# In[ ]:


# Create a heatmap with Seaborn
plt.figure(figsize=(12, 8))  # Set the figure size

# Create the heatmap with tilted x-axis labels
sns.heatmap(df_TS, cmap="Reds", xticklabels=df_TS.columns, yticklabels=df_TS.index,
            vmin=.25, vmax=1)

# Tilt x-axis labels for better readability
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels by 45 degrees for readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.tight_layout()


#%%

# # In[ ]:


# # Create the heatmap with tilted x-axis labels
# plt.figure()
# sns.heatmap(corr_array, xticklabels=corr_Xtr.keys(), yticklabels=corr_Xtr.keys(), cmap="RdBu_r")

# plt.xticks(rotation=90, ha="right")  # Rotate x-axis labels by 45 degrees for readability
# plt.yticks(rotation=0)  # Keep y-axis labels horizontal


# #plt.subplot(212)
# #sns.heatmap(cosine_array, xticklabels=corr_Xtr.keys(), yticklabels=corr_Xtr.keys(), cmap="RdBu")
# #plt.tight_layout()


# # In[ ]:





# # In[ ]:




# # hist_data = [histogram_step_points(data[x_i, y_i, :], bins="auto", density=True) 
# #         for x_i in row_order for y_i in col_order]
# # mean_data = np.reshape(mean_xy_data[row_order][col_order], -1)

# # x, y = hist_data[0].T[0], np.zeros(hist_data[0].shape[0]) # initial hist is all at 0



# # In[ ]:


# # accuracy plot 

# plt.figure()

# plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
# plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False


# plt.bar(range(len(sorted_accs)), sorted_accs)
# plt.ylabel('balanced accuracy')
# plt.tick_params(labelbottom = False, bottom = False) 




    
# try on parcel-defined scaling
# X_train, X_test, Y_train, Y_test, subjID_train, subjID_test = cat_subjs_train_test_ParcelScale(infold, strtsubj=0, endsubj=29, 
#                                      


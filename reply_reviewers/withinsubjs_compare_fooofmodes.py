# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% importing

# general
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # to deal with chained assignement wantings coming from columns concatenation
import sys
import os
# import dask
from datetime import datetime

#%% custom functions

# setting path for mv_python_utils
sys.path.append('../helper_functions')
from mv_python_utils import loadmat_struct

#%% Pipeline object definition

# functions for PCA/SVM
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# pipeline definer
from sklearn.pipeline import Pipeline

# for inserting tanh in pipeline
from sklearn.preprocessing import FunctionTransformer

# crossvalidation
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

# imputer
from sklearn.impute import SimpleImputer

# scaler
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer # to be used at some point?
from sklearn import metrics

# various 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# define the pipeline to be used
class_pipeline = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, 
                                                            strategy='mean')),
                           ('scaler', RobustScaler()),
                           ('squeezer', FunctionTransformer(np.tanh)),
                           ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                           ('SVM', SVC(C=10))
                           ])



#%% folder(s) definition

#input folder 
infold = '../STRG_computed_features/rev_reply_MEG/aperiodic_only/'

fooof_types = ['original fooof', 'custom']

# output folder
outfold = '../STRG_decoding_accuracy/rev_reply_MEG/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

    
#%%

expconds = ['ECEO', 'VS']
mdltypes = ['TimeFeats']
n_cvs = 10 

fields = ['subjID', 'model', 'condition', 'aperiodic type', 'feature', 'accuracy']
LF_dict = {ifield : [] for ifield in fields}  

for isubj in range(29):
    
    for icond in expconds:
        
        for imdl in mdltypes:
                        
            fname = infold + f'{isubj+1:02d}_{icond}_{imdl}_FOOOFcompare.mat'

            mat_content = loadmat_struct(fname)
            F = mat_content['variableName']
            Y_labels = F['Y']
            single_feats = F['single_feats']

                        
            X_dict_fooof = {'offset' : single_feats['fooof_offset'],
                            'slope' : single_feats['fooof_slope'],
                            'both parameters' : np.concatenate((single_feats['fooof_offset'], single_feats['fooof_slope']), axis=1)}
                                                
            X_dict_custom = {'offset' : single_feats['custom_offset'],
                             'slope' : single_feats['custom_slope'],
                             'both parameters' : np.concatenate((single_feats['custom_offset'], single_feats['custom_slope']), axis=1)}

            groups = KBinsDiscretizer(n_bins=n_cvs, 
                                      encode='ordinal', 
                                      strategy='uniform').fit_transform(F['trl_order'][:,np.newaxis])[:, 0]

            aperiodic_type = ['custom', 'fooof']
            X_list = [X_dict_custom, X_dict_fooof]

            aper_type_counter = 0
            for X in X_list:
                
                aper_type = aperiodic_type[aper_type_counter]
                
                for key, value in X.items():
                    
                    acc = cross_val_score(class_pipeline, value, Y_labels, cv=LeaveOneGroupOut(), 
                                          groups=groups,
                                          scoring='balanced_accuracy').mean()        
                    
                    LF_dict['subjID'].append(f'S{isubj+1:02d}')
                    LF_dict['model'].append(imdl)
                    LF_dict['condition'].append(icond)
                    LF_dict['aperiodic type'].append(aper_type)
                    LF_dict['feature'].append(key)
                    LF_dict['accuracy'].append(acc)
                
                aper_type_counter  += 1
            
        print(f'{isubj+1:02d} {icond} {imdl}')
            
DF_aperiodic = pd.DataFrame().from_dict(LF_dict)
DF_aperiodic.to_csv(outfold + 'WS_aperiodic_compare.csv')

#%% plots and stats

plt.figure()

# ECEO
DF_ECEO = DF_aperiodic.loc[DF_aperiodic['condition']=='ECEO', :]

plt.subplot(121)
sns.violinplot(data=DF_ECEO, y='accuracy', x='feature',
               fill=False,
               split=True, hue='aperiodic type', cut=0)
plt.ylim([.5, 1])

plt.title('EC/EO classification', fontsize=16)

# ECEO
DF_VS = DF_aperiodic.loc[DF_aperiodic['condition']=='VS', :]

plt.subplot(122)
sns.violinplot(data=DF_VS, y='accuracy', x='feature',
               fill=False,
               split=True, hue='aperiodic type', cut=0)
plt.ylim([.5, 1])

plt.title('Bsl/VS classification', fontsize=16)

plt.legend([],[], frameon=False)
plt.tight_layout()




















'''
The script must be paired with a corresponding bash file to call it recursively
Also, it needs to be executed after HPC_freqbands_classifier
EB
'''


# import
import sys
import pickle
import os
import numpy as np
import pandas as pd
import dask

# classification tools
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
# crossvalidation
from sklearn.model_selection import cross_val_score
# pipeline definer
from sklearn.pipeline import Pipeline
# imputer
from sklearn.impute import SimpleImputer
# scaler
from sklearn.preprocessing import RobustScaler
# shuffler (before SVC, for mitigating across subjects differences in training?)
from sklearn.utils import shuffle


# current freq band
print(sys.argv)
iBPcond = sys.argv[1]
print('extracting ' + iBPcond)

# input folder
infold = '../STRG_computed_features/'

# output folder
outfold = '../STRG_decoding_accuracy/'
if not(os.path.isdir(outfold)):
    os.mkdir(outfold)

# define parallel function
@dask.delayed
def par_classify(X_train, Y_train, X_test, Y_test, featname):

    # give feedback on process started
    print(featname + ' is getting computed')

    # define the classification pipeline
    # definition necessary here to avoid weird overhead bugs, where models of one feature were tested (or at least,
    # prepared in size preallocation) for a parallel feature
    pipe = Pipeline([('inpute_missing', SimpleImputer(missing_values=np.nan, strategy='mean')),
                     ('scaler', RobustScaler()),
                     ('std_PCA', PCA(n_components=.9, svd_solver='full')),
                     ('SVM', SVC(C=10))
                    ])

    # shuffle train and test
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

    # first obtain crossvalidated accuracy on the training test
    try:
        CV_acc = cross_val_score(pipe, X_train, Y_train, cv=10,
                              scoring='balanced_accuracy').mean()
    except:
        CV_acc = np.nan


    try:
        # train full model
        full_mdl = pipe.fit(X_train, Y_train)

        # generalize and evaluate generalization accuracy
        test_preds_Y = full_mdl.predict(X_test)
        GEN_acc = balanced_accuracy_score(Y_test, test_preds_Y)

    except:

        foo = 'moo'
        GEN_acc = np.nan

    # return a DF: col>feat name; rows>CV_accuracy, GEN_accuracy
    rownames = ['CV_accuracy', 'GEN_accuracy']
    DF = pd.DataFrame(data=[CV_acc, GEN_acc], columns=[featname], index=rownames)

    return DF


# fnames for definition
fname_Xtrain = infold + 'train_X_merged_' + iBPcond + '.pickle'
fname_Ytrain = infold + 'train_Y_merged_' + iBPcond + '.pickle'
fname_Xtest = infold + 'test_X_merged_' + iBPcond + '.pickle'
fname_Ytest = infold + 'test_Y_merged_' + iBPcond + '.pickle'

# load pickles
with open(fname_Xtrain, 'rb') as handle:
    X_train = pickle.load(handle)
with open(fname_Ytrain, 'rb') as handle:
    Y_train = pickle.load(handle)
with open(fname_Xtest, 'rb') as handle:
    X_test = pickle.load(handle)
with open(fname_Ytest, 'rb') as handle:
    Y_test = pickle.load(handle)

# select only aggregate
nparcels = X_train['std'].shape[1]
X_train_aggregate = X_train['aggregate']
X_test_aggregate = X_test['aggregate']

# loop across parcels
parcels_list_DF = []

for iparcel in range(nparcels):

    this_parcel_idxs = np.arange(iparcel, X_train_aggregate.shape[1], nparcels, dtype='int')
    X_parc_feats_train = X_train_aggregate[:, this_parcel_idxs]
    X_leftout = X_test_aggregate[:, this_parcel_idxs]

    DF = par_classify(X_parc_feats_train, Y_train, X_leftout, Y_test, str(iparcel))
    parcels_list_DF.append(DF)

    print(iparcel)

# actually launch the process
out_DFs = dask.compute(parcels_list_DF)
ordered_accs_DF = pd.concat(out_DFs[0], axis=1)

# save
fname_out = '../STRG_decoding_accuracy/' + iBPcond + '_parcels_accs.csv'
ordered_accs_DF.to_csv(fname_out)


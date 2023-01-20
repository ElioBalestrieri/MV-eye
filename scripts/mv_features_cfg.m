function cfg_feats = mv_features_cfg()
% cfg definition for MV features

% time domain
cfg_feats.time = {'mean','median','std','SAMPEN', 'wpH'};

% frequency domain
cfg_feats.freqs = {'alphaPow'}; % placeholder, currently this field isnot used.

% PCA var explained
cfg_feats.PCAvarExplained = .9;

% classifiers compared
cfg_feats.classifiers = {'SVM', 'LDA', 'NaiveBayes'};
cfg_feats.kFoldNum = 10;

% choose only one classifier to be trained on all (360!) parcels
cfg_feats.parcelClass = 'SVM';

% verbosity
cfg_feats.verbose = true; 


end
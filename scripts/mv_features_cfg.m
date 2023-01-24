function cfg_feats = mv_features_cfg()
% cfg definition for MV features

% time domain
cfg_feats.time = {'mean','median','std', 'SAMPEN'}; %, 'wpH'};

% frequency domain
cfg_feats.freq = {'covFFT'}; % placeholder, currently this field isnot used.

% choose only one classifier to be trained on all (360!) parcels
cfg_feats.parcelClass = 'SVM'; % options: 'SVM', 'LDA', 'NaiveBayes'
cfg_feats.kFoldNum = 10;

% verbosity
cfg_feats.verbose = true; 


end
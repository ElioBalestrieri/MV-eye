function cfg_feat = mv_features_cfg()
% cfg definition for MV features

% time domain
cfg_feat.time = {'SAMPEN'};

% frequency domain
cfg_feat.freqs = {'alphaPow'};

% space dimensionality reduction
cfg_feat.keepchans = false;


end
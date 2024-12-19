%% EC/EO decoding
clearvars;
close all
clc

% start timing

%% path definition

% input and packages
software_path           = '/home/balestrieri/sciebo/Software/biosig-master/biosig/t300_FeatureExtraction';
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '../STRG_data/MPI_LEMON_ECEO';
entropy_functions_path  = '../../Software/mMSE-master';
helper_functions_path   = '../helper_functions/';
beta_functions_path     = '../beta_functions';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '/home/balestrieri/sciebo/Software/catch22/wrap_Matlab';

% output folders
out_feats_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/LEMON';
if ~isfolder(out_feats_path); mkdir(out_feats_path); end

addpath(software_path); addpath(helper_functions_path)
addpath(entropy_functions_path); addpath(resources_path)
addpath(plotting_functions_path); 
addpath(catch22_path)
addpath(fieldtrip_path); 
ft_defaults

% LEMON data is in set format. Load EEGLAB to open files
eeglab nogui

% fetch subj names
subjects_EC = dir(fullfile(data_path, 'LEMON-closed-preprocessed', 'sub-0*'));
subjects_EO = dir(fullfile(data_path, 'LEMON-open-preprocessed', 'sub-0*'));

matching_subjs = ismember({subjects_EO.name}, {subjects_EC.name});
subjects_EO = subjects_EO(matching_subjs);


isubj = 1;
fname = [subjects_EC(isubj).name, '_eeg.set']; 
pathtofile_EC = fullfile(subjects_EC(isubj).folder, subjects_EC(isubj).name, 'eeg');
pathtofile_EO = fullfile(subjects_EO(isubj).folder, subjects_EC(isubj).name, 'eeg');

EC_eeg = pop_loadset('filename', fname, 'filepath', pathtofile_EC);
EO_eeg = pop_loadset('filename', fname, 'filepath', pathtofile_EO);
dat = {EC_eeg, EO_eeg}; 

% list of channels considered (POSTERIOR)
KEEPCHANS = {'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', ...
             'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', ...
             'PO9', 'PO7', 'PO3', 'PO4', 'PO8', 'PO10', 'O1', 'O2', ...
             'CPz', 'Pz', 'POz', 'Oz'};

% plot channels
tmp = struct2table(EO_eeg.chanlocs); chanlabs = tmp.labels;
idxs_chans = find(ismember(chanlabs, KEEPCHANS));
figure(); 
topoplot(-ones(61, 1), EO_eeg.chanlocs, 'emarker2', {idxs_chans, 'o', 'w', 5})
clim([-2, 2])

% convert to fieldtrip structure
dat = cellfun(@(x) eeglab2fieldtrip(x, 'raw', 'none'), dat, 'UniformOutput',false);

% select only posterior channels 
cfg = []; cfg.channel = KEEPCHANS;
dat = cellfun(@(x) ft_preprocessing(cfg, x), dat, 'UniformOutput',false);

% cut in 1 s segments
for idat = 1:2

    % redefine trials in 1 sec segments
    cfg_pre = [];
    cfg_pre.toilim = [0, 1-1./dat{idat}.fsample];
    mini1 = ft_redefinetrial(cfg_pre, dat{idat});
    
    cfg_post = [];
    cfg_post.toilim = [1, 2-1./dat{idat}.fsample];
    mini2 = ft_redefinetrial(cfg_post, dat{idat});

    cfg = [];
    segmented_1s = ft_appenddata(cfg, mini1, mini2);

    dat{idat} = segmented_1s;

end

%% merge datasets & set config
% unitary label
Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';
% merge
cfg = [];
dat = ft_appenddata(cfg, dat{1}, dat{2});
% call for config
cfg_feats = mv_features_cfg();

% shuffle
rng(1)
shffld_idxs = randperm(length(Y));
Y = Y(shffld_idxs);
dat.trial = dat.trial(shffld_idxs);

%% initialize Feature structure
ntrials = length(dat.trial);

F.single_parcels = [];
F.single_feats = [];
F.multi_feats = double.empty(ntrials, 0);
F.Y = Y;

%% compute frequency features

F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);

%% compute time features

F = mv_features_timedomain(cfg_feats, dat, F);

%% compute catch22

F = mv_wrap_catch22(cfg_feats, dat, F);

%% double check BIDS pinfo between sessions (and merge them if equal)
for idx = 1:numel(EC_eeg.BIDS.pInfo)
    tests_match(idx) = all(EC_eeg.BIDS.pInfo{idx} == EO_eeg.BIDS.pInfo{idx}); %#ok<SAGROW> 
end
sessions_match = all(tests_match);

if sessions_match % store the pInfo in F
    F.pinfo = EO_eeg.BIDS.pInfo; 
else % completely null the F structure computed so far, in order to prevent 
     % mistakes at further stages
    F = [];
    F.errormessage = 'mismatch in BIDS';
    fname = ERROR;
end


%% store F output
% decoding will continue in python

% save
fname = 'test.mat';
save(fullfile(out_feats_path, fname), 'F')

toc

%%

% % fetch matrices of bad channels
% badchans_EC = load(fullfile(subjects_EC(1).folder, 'preprocessing_visualization', 'badchans.mat'));
% badchans_EO = load(fullfile(subjects_EO(1).folder, 'preprocessing_visualization', 'badchans.mat'));
% 
% % erase exceedign subject from EO
% badchans_EO.badchans = badchans_EO.badchans(:, matching_subjs);
% badchans_EO.s_ids = badchans_EO.s_ids(matching_subjs);
% 
% % check that all chans and subjs are aligned
% test_ids = all(strcmp(badchans_EO.s_ids, badchans_EC.s_ids));
% test_chans = all(strcmp(badchans_EO.chanlabels, badchans_EC.chanlabels));
% if ~(test_chans && test_ids); error('NEIN NEIN NEIN!'); end
% 
% % create unitary matrix for bad channels in both conditions. They will be
% % removed from both conds
% badchans_EO = array2table(badchans_EO.badchans, ...
%                           'RowNames', badchans_EO.chanlabels, ...
%                           'VariableNames', badchans_EO.s_ids);
% badchans_EC = array2table(badchans_EC.badchans, ...
%                           'RowNames', badchans_EC.chanlabels, ...
%                           'VariableNames', badchans_EC.s_ids);
% badchans_EO = badchans_EO(:, matching_subjs);


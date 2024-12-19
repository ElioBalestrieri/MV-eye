%% VG features extraction
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '../STRG_data/MPI_LEMON_ECEO';
helper_functions_path   = '../helper_functions/';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '/home/balestrieri/sciebo/Software/catch22/wrap_Matlab';

out_feat_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/LEMON';
if ~isfolder(out_feat_path); mkdir(out_feat_path); end

addpath(helper_functions_path)
addpath(plotting_functions_path); addpath(resources_path)
addpath(catch22_path)
addpath(fieldtrip_path); 

%% prepare data fetching

% fetch subj names
subjects_EC = dir(fullfile(data_path, 'LEMON-closed-preprocessed', 'sub-0*'));
subjects_EO = dir(fullfile(data_path, 'LEMON-open-preprocessed', 'sub-0*'));

% reminder: use only EC subject list for fetching the filenames also in the
% EO condition. this checks automatically then for potential mismatches...
% Also EC is the condition with one subject less.
matching_subjs = ismember({subjects_EO.name}, {subjects_EC.name});
subjects_EO = subjects_EO(matching_subjs);

nsubjs = length(subjects_EO);

% list of channels considered (POSTERIOR)
KEEPCHANS = {'POz'};

% loop through "theoretical" models
mdls_codes = {'gammaCompare'}; % {'FreqBands', 'FullFFT', 'TimeFeats'};


%% loop into subjects

% LEMON data is in set format. Load EEGLAB to open files
eeglab nogui

for isubj = 100

    % weird errors if ft called outside parfor
    ft_defaults;

    % collect data and concatenate in cell
    subjcode = subjects_EC(isubj).name; 
    fname = [subjcode, '_eeg.set']; 
    pathtofile_EC = fullfile(subjects_EC(isubj).folder, subjects_EC(isubj).name, 'eeg');
    pathtofile_EO = fullfile(subjects_EO(isubj).folder, subjects_EC(isubj).name, 'eeg');
    
    EC_eeg = pop_loadset('filename', fname, 'filepath', pathtofile_EC);
    EO_eeg = pop_loadset('filename', fname, 'filepath', pathtofile_EO);
    dat = {EC_eeg, EO_eeg}; 

    % convert to fieldtrip structure
    dat = cellfun(@(x) eeglab2fieldtrip(x, 'raw', 'none'), dat, 'UniformOutput',false);
    
    % select only posterior channels 
    cfg = []; cfg.channel = KEEPCHANS;
    dat = cellfun(@(x) ft_preprocessing(cfg, x), dat, 'UniformOutput',false);
    
    % feedback
    fprintf('\n\n######################\n')
    fprintf('Subj %s completed\n', subjcode)

end

%% get reduced data

EC_dat = cat(2, dat{1}.trial{:})';
EO_dat = cat(2, dat{2}.trial{:})';

save("EC_dat.mat", 'EC_dat')
save("EO_dat.mat", 'EO_dat')






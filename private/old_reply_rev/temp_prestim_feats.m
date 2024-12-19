function temp_prestim_feats(nthreads)

% add necessary packages

fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
addpath(fieldtrip_path)

catch22_path            = '/home/balestrieri/sciebo/Software/catch22/wrap_Matlab'; %'/home/e/ebalestr/toolboxes/catch22/wrap_Matlab';
addpath(catch22_path)

helper_functions_path   = '../helper_functions/';
addpath(helper_functions_path)

% define input/output folder
outdir = '../STRG_computed_features/NNprestim';
indir = '/remotedata/AgGross/AlphaSpade/Data/_meg'; % '/scratch/tmp/grossjoa/AlphaSpade/Raw';
indir_beh = '/remotedata/AgGross/AlphaSpade/Data/_behav';

if ~isfolder(outdir)
    mkdir(outdir)
end

% define subject list 
filelist = struct2table(dir(fullfile(indir , 'fullsegcleanmeg_*.mat'))); % '/remotedata/AgGross/AlphaSpade/Data/_meg'
filenames = filelist.name;

% extract participants codes
pat= regexpPattern('\d');
codes_pre = cellfun(@(x) extract(x, pat), filenames, 'UniformOutput', false);

for iPre = 1:length(codes_pre)
    codes_pre{iPre} = [codes_pre{iPre}{:}];
end

%% blabla
% nsubjs = length(fnames_sources);
% thisObj = parpool(nthreads); 

n_subjs = length(filenames);

ft_defaults % to be put maybe in the loop?

[fnames_sources, fnames_targets] = deal(cell(n_subjs, 1));

for isubj = 1:n_subjs 
    
    this_fname = filenames{isubj}; subjcode = this_fname(17:20);
    temp = load(fullfile(indir, this_fname));
    indat = temp.fullsegdata;

    % load behavioral datafile
    beh = load(fullfile(indir_beh, ['fullphaselock2_', subjcode, '.mat']));

    % append fine grained info on  stimulus and response to the trialinfo
    indat.trialinfo(:, end+1) = beh.phaselock(:, 5);
    indat.trialinfo(:, end+1) = 10.^beh.phaselock(:, 8);
    
    % prestimulus selection
    cfg = [];
    cfg.latency = [-1.2, -.2];
    cfg.trials = find(indat.trialinfo(:, end)~=1);
    prestim = ft_selectdata(cfg, indat);
    ntrials = length(prestim.trialinfo);

    % lpfreq
    cfg = [];
    cfg.lpfilter = 'yes';
    cfg.lpfreq = 100;
    prestim = ft_preprocessing(cfg, prestim);

    % resample
    cfg = [];
    cfg.resamplefs = 256;
    prestim = ft_resampledata(cfg, prestim);

    % scale up data  
    prestim.trial = cellfun(@(x) x*1e11, prestim.trial, 'UniformOutput',false);

    % call for config
    F.single_parcels = [];
    F.single_feats = [];
    F.multi_feats = double.empty(ntrials, 0);
    F.Y = prestim.trialinfo(:, end-1);
    F.x = zscore(prestim.trialinfo(:, end));
    F.trl_order = 1:ntrials;


    cfg_feats = mv_features_cfg();
    F = mv_features_timedomain(cfg_feats, prestim, F);
    F = mv_wrap_catch22(cfg_feats, prestim, F);
    F = mv_periodic_aperiodic(cfg_feats, prestim, F);
    F.cfg_timefeats = cfg_feats;

    % further freq bands to the same structure
    cfg_feats = mv_features_cfg_theoretical_freqbands();
    F = mv_features_freqdomain_nonrecursive(cfg_feats, prestim, F);

    F.trialinfo = prestim.trialinfo;

    save(fullfile(outdir, [num2str(isubj), '.mat']), "F")

end


end




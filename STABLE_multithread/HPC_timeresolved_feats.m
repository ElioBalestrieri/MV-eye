function HPC_timeresolved_feats(nthreads)

% add necessary packages
addpath('/home/e/ebalestr/toolboxes/fieldtrip-20221223')

catch22_path = '/home/e/ebalestr/toolboxes/catch22/wrap_Matlab';
addpath(catch22_path)

helper_functions_path   = '../helper_functions/';
addpath(helper_functions_path)

% define input/output folder
outdir_temp = '../STRG_data/alphaSPADE/segmented';
outdir = '../STRG_computed_features/alphaSPADE/segmented';
indir = '/scratch/tmp/grossjoa/AlphaSpade/Raw';

if ~isfolder(outdir_temp)
    mkdir(outdir_temp)
end

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

%% segment data and store it on disk
% in order to prevent memory fail. Hence to be performed serially
n_subjs = length(filenames); n_parts = 10;

ft_defaults

[fnames_sources, fnames_targets] = deal(cell(n_subjs, 1));

for isubj = 1:n_subjs 
    
    this_fname = filenames{isubj}; subjcode = this_fname(17:20);
    temp = load(fullfile(indir, this_fname));
    indat = temp.fullsegdata;
    
    % for the current subject, generate root destination folder if it does
    % not exist already
    root_subjSource_fold = fullfile(outdir_temp, subjcode);
    
    if ~isfolder(root_subjSource_fold)
        mkdir(root_subjSource_fold)
    end
    
    % repeat also for the target folder
    root_subjTarget_fold = fullfile(outdir, subjcode);

    if ~isfolder(root_subjTarget_fold)
        mkdir(root_subjTarget_fold)
    end

    n_trls = length(indat.trial); 
    
    leftouts = ceil(n_trls/n_parts)*n_parts-n_trls;
    trial_nums = [1:n_trls, nan(1, leftouts)]; ntrlsXpart = length(trial_nums)/n_parts;
    partitioned_trials = reshape(trial_nums, ntrlsXpart, n_parts);
    
    acc_ = 1;
    for iTrialSet = 1:n_parts
        
        these_trls = partitioned_trials(:, iTrialSet);
        these_trls = these_trls(~isnan(these_trls));
        
        try
        
            cfg = [];
            cfg.trials = these_trls;
            small_dat = ft_selectdata(cfg, indat);
            small_dat.trialsequence = these_trls;

            fname_source = fullfile(root_subjSource_fold, [subjcode, '_part', num2str(iTrialSet), '.mat']);        
            save(fname_source, 'small_dat')

            % append fnames to both source and target
            fname_target = fullfile(root_subjTarget_fold, [subjcode, '_part', num2str(iTrialSet), '.mat']);
            cell_fnames_S{acc_} = fname_source;
            cell_fnames_T{acc_} = fname_target;
            
            acc_ = acc_ + 1;
            
        catch
            
            fprintf('\nFailed trial segmentation for subj %s', subjcode)
          
        end
        
        
        
    end
        
    fnames_sources{isubj} = cell_fnames_S;
    fnames_targets{isubj} = cell_fnames_T;
        
end


%% prepare subject number and thread parfor 
nsubjs = length(fnames_sources);
thisObj = parpool(nthreads); 

for isubj = 1:nsubjs 

    ft_defaults;
    fnames_S = fnames_sources{isubj}; fnames_T = fnames_targets{isubj};
    n_files = length(fnames_S);
    
    % already define here moving window, constant for all files
    cfg = [];
    cfg.stepsize = .01; 
    cfg.winsize = .25;

    for ifile = 1:n_files
        
        this_fname_S = fnames_S{ifile};
        this_fname_T = fnames_T{ifile};        
        temp = load(this_fname_S);
        indat = temp.small_dat;
    
        % scale up data  
        indat.trial = cellfun(@(x) x*1e11, indat.trial, 'UniformOutput',false);    
        outdat = mv_timeresolved(cfg, indat);
        outdat.trialsequence = indat.trialsequence;

        saveinparfor(this_fname_T, outdat)
        
    end

end

delete(thisObj) 

end



% old chunks
% define subject list already computed
% already_comp_subjs = struct2table(dir(fullfile(outdir , '*_timeresolved.mat'))); % '/remotedata/AgGross/AlphaSpade/Data/_meg'
% already_comp_subjs = already_comp_subjs.name;
% 
% % exclude all the subjects that have already been computed
% pat= regexpPattern('\d');
% 
% codes_pre = cellfun(@(x) extract(x, pat), filenames, 'UniformOutput', false);
% codes_post = cellfun(@(x) extract(x, pat), already_comp_subjs, 'UniformOutput', false);
% 
% for iPre = 1:length(codes_pre)
%     codes_pre{iPre} = [codes_pre{iPre}{:}];
% end
% 
% for iPost = 1:length(codes_post)
%     codes_post{iPost} = [codes_post{iPost}{:}];
% end
% 
% already_comp_positions = ismember(codes_pre, codes_post);
% filenames = filenames(~already_comp_positions);



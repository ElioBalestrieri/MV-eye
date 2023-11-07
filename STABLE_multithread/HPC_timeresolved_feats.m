function HPC_timeresolved_feats(nthreads)

% add necessary packages
addpath('/home/e/ebalestr/toolboxes/fieldtrip-20221223')

catch22_path = '/home/e/ebalestr/toolboxes/catch22/wrap_Matlab';
addpath(catch22_path)
helper_functions_path   = '../helper_functions/';
addpath(helper_functions_path)

% define input/output folder
outdir = '../STRG_computed_features/alphaSPADE';
indir = '/scratch/tmp/grossjoa/AlphaSpade/Raw';

if ~isfolder(outdir)
    mkdir(outdir)
end


% define subject list 
filelist = struct2table(dir(fullfile(indir , 'fullsegcleanmeg_*.mat'))); % '/remotedata/AgGross/AlphaSpade/Data/_meg'
filenames = filelist.name;

% prepare subject number and thread parfor 
nsubjs = length(filenames);
thisObj = parpool(nthreads);

parfor isubj = 1:nsubjs

    ft_defaults;

    this_fname = filenames{isubj}; subjcode = this_fname(17:20);
    temp = load(fullfile(indir, this_fname));
    indat = temp.fullsegdata;

    cfg = [];
    cfg.stepsize = .01;
    cfg.winsize = .25;
    
    % scale up data  
    indat.trial = cellfun(@(x) x*1e11, indat.trial, 'UniformOutput',false);
    
    outdat = mv_timeresolved(cfg, indat);

    fname_out_feat = [subjcode, '_timeresolved.mat']
    saveinparfor(fullfile(outdir, fname_out_feat), outdat)

end

delete(thisObj)

end


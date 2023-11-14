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

% define subject list already computed
already_comp_subjs = struct2table(dir(fullfile(outdir , '*_timeresolved.mat'))); % '/remotedata/AgGross/AlphaSpade/Data/_meg'
already_comp_subjs = already_comp_subjs.name;

% exclude all the subjects that have already been computed
pat= regexpPattern('\d');

codes_pre = cellfun(@(x) extract(x, pat), filenames, 'UniformOutput', false);
codes_post = cellfun(@(x) extract(x, pat), already_comp_subjs, 'UniformOutput', false);

for iPre = 1:length(codes_pre)
    codes_pre{iPre} = [codes_pre{iPre}{:}];
end

for iPost = 1:length(codes_post)
    codes_post{iPost} = [codes_post{iPost}{:}];
end

already_comp_positions = ismember(codes_pre, codes_post);
filenames = filenames(~already_comp_positions);

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


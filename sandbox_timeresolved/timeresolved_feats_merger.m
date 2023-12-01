clearvars
close all
clc


allsubjsdir = '../STRG_computed_features/alphaSPADE/segmented/';

% define subject list 
folds = dir(allsubjsdir); subjs = {folds(3:end).name}; % skip . and ..

for isubj = subjs

    % preamble: input, output, fnames
    ID = isubj{1};
    source_dir = fullfile(allsubjsdir, ID);
    target_dir_feats = fullfile('../STRG_computed_features/alphaSPADE', 'timeresolved_split_feats', ID);
    if ~isfolder(target_dir_feats); mkdir(target_dir_feats); end
    fnames = struct2table(dir(fullfile(source_dir, '*part*.mat'))); 
    fnames = fnames.name'; 

    for ifile = 1:length(fnames)

        this_file = fnames{ifile};
        temp = load(fullfile(source_dir, this_file));

        F = temp.variableName;
        F.single_feats = structfun(@(x) single(x), F.single_feats, 'UniformOutput',false);

        if ifile == 1

            HDR = [];
            HDR.time_winCENter = F.time_winCENter;
            HDR.time_winONset = F.time_winONset;
            HDR.time_winOFFset = F.time_winOFFset;
            HDR.label = F.label;
            HDR.featnames = fieldnames(F.single_feats)';

            save(fullfile(target_dir_feats, [ID, '_HDR.mat']), "HDR");

            ACC = F.single_feats;
            rawY = F.trialinfo;

        else

            for ifeat = HDR.featnames
                ACC.(ifeat{1}) = cat(1, ACC.(ifeat{1}), F.single_feats.(ifeat{1}));
            end

            rawY = [rawY; F.trialinfo]; %#ok<AGROW> 

        end

        fprintf('\nFile %i/%i', ifile, length(fnames))

    end

    for ifeat = HDR.featnames

        Decode = [];
        Decode.X = ACC.(ifeat{1});
        Decode.rawY = rawY;

        save(fullfile(target_dir_feats, [ID, '_', ifeat{1} '.mat']), 'Decode')

        fprintf('%s; ', ifeat{1})

    end

    fprintf('\n\n#################################')
    fprintf('\nFinished with subj %i/%i', isubj{1}, length(subjs))
    

end

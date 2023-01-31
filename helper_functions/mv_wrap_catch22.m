function F = mv_wrap_catch22(cfg_feats, dat, F)

% give the opportunity to not run it if specified 
if isfield(cfg_feats, 'catch22flag')
    if ~cfg_feats.catch22flag
        return
    end
else
    warning('catch22 field not specified, and not computing it')
    return
end


% vars needed for future loops
nchans = length(dat.label);
ntrials = length(dat.trial);

%% chan, trial nested loop

inCount = 0; totcount = ntrials*nchans; nsteps = 21;
stepsFeedback = round(linspace(1, totcount, nsteps));
percentFinished = round(100*stepsFeedback/totcount);

catch22FeatMat = nan(ntrials, 22, nchans);

tic
for itrl = 1:ntrials

    this_trl = dat.trial{itrl};

    for ichan = 1:nchans

        tvector = this_trl(ichan, :);
        [out22, names] = catch22_all(tvector', false);    
        
        catch22FeatMat(itrl, :, ichan) = out22;

        if any(inCount==stepsFeedback) && cfg_feats.verbose
            fprintf('\n\n\n\n\n\n\n\n\n')
            fprintf('%i%% of catch22 computation completed\n\n', ...
                    percentFinished(inCount==stepsFeedback))
        end

        inCount = inCount+1;

    end

end

%% 
for ifeat = 1:length(names)

    this_feat = names{ifeat}; 
    
    % fetch TEMP from the big matrix of features precomputed
    TEMP = squeeze(catch22FeatMat(:, ifeat, :));

    % reduce from parcels to PC
    F.single_feats.(this_feat) = TEMP;
    % add to F structure
    F = local_add_feature(F, TEMP, ntrials, nchans);

end

% log runtime
F.runtime.catch22 = round(toc, 2);



end

%% ########################### LOCAL FUNCTIONS ############################

function F = local_add_feature(F, origFeat, ntrials, nchans)

% single parcels (or chans) 
if isempty(F.single_parcels)
    % create a subfield of the dat structure storing features 
    F.single_parcels = mat2cell(origFeat, ntrials, ones(nchans, 1));
else
    for iparc = 1:length(F.single_parcels)
        F.single_parcels{iparc} = [F.single_parcels{iparc}, origFeat(:,iparc)];
    end
end

end


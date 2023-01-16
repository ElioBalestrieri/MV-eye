function dat = mv_features_timedomain(cfg_feats, dat)

% quick input check
if ~isfield(cfg_feats, 'time')
    disp('No features in the time domain')
    return
else
    if isempty(cfg_feats.time)
        disp('No features in the time domain')
        return
    end
end

% vars needed for future loops
nchans = length(dat.label);
ntrials = length(dat.trial);

% initialize Features structure
F = struct();
F.mv = double.empty(ntrials, 0);

% initiate switch loop for features
for ifeat = cfg_feats.time

    switch ifeat{1}

        case 'SAMPEN' % SAMPle ENtropy     

            F.sampen = zeros(ntrials, nchans);
            for itrl = 1:ntrials
                F.sampen(itrl, :) = SampEnMat(2, .2, ...
                    zscore(dat.trial{itrl}, [], 2));    
            end

            if ~cfg_feats.keepchans

                % [F.mv(:, end+1), w] = local_PCA(F.sampen);
                [~, PCAscore] = pca(F.sampen);
                F.temp = PCAscore(:, 1:3);

            end

        otherwise
            error('"%s" is not recognized as feature', ifeat{1})

    end

end

dat.F = F;

end

%% ########################### LOCAL FUNCTIONS ############################


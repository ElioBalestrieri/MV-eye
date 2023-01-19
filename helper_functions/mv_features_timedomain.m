function F = mv_features_timedomain(cfg_feats, dat, F)

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

% initiate switch loop for features
for ifeat = cfg_feats.time

    switch ifeat{1}

        case 'SAMPEN' % SAMPle ENtropy     

            sampen = zeros(ntrials, nchans);
            for itrl = 1:ntrials
                sampen(itrl, :) = SampEnMat(2, .2, ...
                    zscore(dat.trial{itrl}, [], 2));    
            end

            [~, PCAscore, ~, ~, EXPLAINED] = pca(sampen);
            keepcomps = (cumsum(EXPLAINED)/100) <= cfg_feats.PCAvarExplained;

            F.single_feats.sampen = PCAscore(:, keepcomps);
             
            % only first component for multifeats
            F.multi_feats(:, end+1) = PCAscore(:, 1);

            % single parcels (or chans) 
            if isempty(F.single_parcels)
                % create a subfield of the dat structure storing features 
                F.single_parcels = mat2cell(sampen, ntrials, ones(nchans, 1));
            else
                for iparc = 1:length(F.single_parcels)
                    F.single_parcels{iparc} = [F.single_parcels{iparc}, sampen(:,iparc)];
                end
            end




        otherwise
            error('"%s" is not recognized as feature', ifeat{1})

    end

end

end

%% ########################### LOCAL FUNCTIONS ############################


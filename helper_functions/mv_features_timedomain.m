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

% note: since many features will be computed, I decided to use only one
% variable, named "TEMP", to store all the temporary computations for
% features. This will spare quite some matrices in memory, which in
% parallel computing could be relevant.
% The data is then assigned to the F structure with its rightful name.

for ifeat = cfg_feats.time

    this_feat = ifeat{1}; 
    
    if cfg_feats.verbose; fprintf('\nComputing %s', this_feat); end

    tic 
    
    % preallocate TEMP (and get rid of previous values, just to be sure)
    TEMP = nan(ntrials, nchans);

    switch this_feat

        case 'SAMPEN' % SAMPle ENtropy     

            for itrl = 1:ntrials
                TEMP(itrl, :) = SampEnMat(2, .2, ...
                    zscore(dat.trial{itrl}, [], 2));    
            end
            
        case 'median'

            TEMP = cellfun(@(x) median(x, 2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';

        case 'mean'

            TEMP = cellfun(@(x) mean(x, 2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';

        case 'std'

            TEMP = cellfun(@(x) std(x,[],2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';

        case 'wpH'

            for itrl = 1:ntrials
                for ichan = 1:nchans
                    TEMP(itrl,ichan) = wpH(zscore(dat.trial{itrl}(ichan,:)),3,1);
                end
            end                    

        case 'kurtosis'

            TEMP = cellfun(@(x) kurtosis(x,1,2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';            

        case 'skewness'

            TEMP = cellfun(@(x) skewness(x,1,2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';            

        case 'Hurst_exp'

            for itrl = 1:ntrials
                for ichan = 1:nchans
                    TEMP(itrl,ichan) = hurst_exponent(zscore(dat.trial{itrl}(ichan,:)));
                end
            end                    

        case 'zero_cross'

            swap = cellfun(@(x) detrend(x'), dat.trial, 'UniformOutput',false);
            marks = cellfun(@(x) convn(sign(x), ones(2, 1), 'same')', swap, 'UniformOutput',false);
            TEMP = cat(3, marks{:}); TEMP = squeeze(sum(TEMP==0, 2))';

        case 'zero_cross_derivative'

            swap = cellfun(@(x) diff(x'), dat.trial, 'UniformOutput',false);
            marks = cellfun(@(x) convn(sign(x), ones(2, 1), 'same')', swap, 'UniformOutput',false);
            TEMP = cat(3, marks{:}); TEMP = squeeze(sum(TEMP==0, 2))';

        case 'hjorth_mobility'

            TEMP = cat(3, dat.trial{:});
            TEMP = local_hjorth_mobility(TEMP);

        case 'hjorth_complexity'

            TEMP = cat(3, dat.trial{:});
            derTEMP = diff(TEMP, 1, 2);

            TEMP = local_hjorth_mobility(derTEMP)./local_hjorth_mobility(TEMP);


        otherwise

            error('"%s" is not recognized as feature', ifeat{1})

    end

    % reduce from parcels to PC
    F.single_feats.(this_feat) = TEMP;
    % add to F structure
    F = local_add_feature(F, TEMP, ntrials, nchans);
    % log runtime
    F.runtime.(this_feat) = round(toc, 2);

end

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


function M = local_hjorth_mobility(inArray)
% takes N dimensional array. time MUST be on the second dimension
% https://en.wikipedia.org/wiki/Hjorth_parameters

den = squeeze(var(inArray, [], 2))';
num = squeeze(var(diff(inArray, 1, 2), [], 2))';
M = sqrt(num./den);

end
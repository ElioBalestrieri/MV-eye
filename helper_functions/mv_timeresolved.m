function outdat = mv_timeresolved(cfg, dat)

% cfg.stepsize (sec)
% cfg.winsize  (sec)

% start time counting
onset_script = tic;

% discretize window. force it odd length, so we have an exact timepoint
% centered
disc_winsize = round(cfg.winsize*dat.fsample);
if mod(disc_winsize, 2)==0
    disc_winsize = disc_winsize-1;
end
md_win = ceil(disc_winsize/2);

% discretize moving step
disc_stepsize = round(cfg.stepsize*dat.fsample);

% while loop to define time windows
outdat = struct('time_winCENter', [], ...
                'time_winONset', [], ...
                'time_winOFFset', []);

% define length of trial, upper bound of the while loop to define the time
% windows.
len_tr = size(dat.trial{1}, 2);

% current structure to be appended to a cell
disc = [];
disc.on_ = 1;
disc.cen_ = md_win;
disc.off_ = disc_winsize;

% preallocate cell & accumulator
acc_ = 0; disc_indexes = {};
while disc.off_<=len_tr

    % update counter
    acc_ = acc_+1;
    disc_indexes{acc_} = disc; %#ok<AGROW> 

    % store current timepoints in the out data structure
    % assuming time constant across all trials
    outdat.time_winONset(acc_) = dat.time{1}(disc.on_);
    outdat.time_winCENter(acc_) = dat.time{1}(disc.cen_);
    outdat.time_winOFFset(acc_) = dat.time{1}(disc.off_);

    % update all disc subfields in one
    disc = structfun(@(x) x+disc_stepsize, disc, 'UniformOutput', false);

end

% append window size, accounting for the rounding and the taking an odd
% number to center the window
outdat.winsize = disc_winsize/dat.fsample;
outdat.stepsize = disc_stepsize/dat.fsample;

% prepare data for computation
nsteps = length(disc_indexes); nchans = length(dat.label); 
ntrls = length(dat.trial);

% define features to be computed based on custom cfg function
% compute feats
cfg_feats = mv_features_cfg();

% loop through windows
for istep = 1:nsteps

    % time window selection
    cfg = [];
    cfg.latency = [outdat.time_winONset(istep), outdat.time_winOFFset(istep)];
    minidat = ft_selectdata(cfg, dat);

    % F structure initialization for custom feature computation
    F = [];
    F.single_parcels = [];
    F.single_feats = [];
    F.multi_feats = double.empty(ntrls, 0);
    
    % actual feature computation
    F = mv_features_timedomain(cfg_feats, minidat, F);
    F = mv_wrap_catch22(cfg_feats, minidat, F);

    % preallocate 3d arrays with features at first step
    if istep == 1
        for ifeat = fieldnames(F.single_feats)
            this_feat = ifeat{1};
            outdat.single_feats.(this_feat) = nan(ntrls, nsteps, nchans);
        end
    end

    % append computed features
    for ifeat = fieldnames(F.single_feats)
        this_feat = ifeat{1};
        outdat.single_feats.(this_feat)(:, istep, :) = F.single_feats.(this_feat); 
    end

end

% append channel names & trialinfo as well, useful for classification &
% interpretation later
outdat.label = dat.label;
outdat.trialinfo = dat.trialinfo;
outdat.full_runtime = toc(onset_script);

end

%% ##################### LOCAL FUNCTIONS ##################################

% function TEMP = local_compute_feature(dat_chunk, this_feat)
% 
% switch this_feat
% 
%     case 'std'
% 
%         TEMP = std(dat_chunk, [], 2);
% 
%     case 'mean'
% 
%         TEMP = mean(dat_chunk, 2);
% 
%     case 'MCL'
% 
%         TEMP = squeeze(mean(abs(diff(dat_chunk, 1, 2).^2), 2));
% 
%     otherwise
% 
%         error('"%s" is not recognized as feature', this_feat)
% 
% end
% 
% TEMP = squeeze(TEMP);
% 
% end

%% old stuff

% dat_mat = cat(3, dat.trial{:});
% 
% for ifeat = 1:length(cfg.features)
%     
%     this_feat = cfg.features{ifeat};
%     outdat.single_feats.(this_feat) = nan(nchans, nsteps, ntrials);
% 
%     for istep = 1:nsteps
% 
%         chunk_definer = disc_indexes{istep};
%         dat_chunk = dat_mat(:, chunk_definer.on_:chunk_definer.off_, :);
% 
%         TEMP = local_compute_feature(dat_chunk, this_feat);
%         outdat.single_feats.(this_feat)(:, istep, :) = TEMP;
% 
%     end
% 
%     fprintf('\n%s completed. (%i/%i)', this_feat, ifeat, length(cfg.features))
% 
% end







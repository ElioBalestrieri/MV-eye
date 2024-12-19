function F = mv_periodic_aperiodic_fooof(cfg_feats, dat, F)

% ignore cfg_feats right now

% same cfg as the FFT
cfg = cfg_feats.cfg_FFT;
cfg.output = 'fooof_aperiodic';
cfg.keeptrials = 'no';

ntrls = length(dat.trial);

%% fooof-based feature extraction

% start logging time from here: loop into each trial to extract the fooof
% pars (no possible to keeptrials :(
tic

feats_mat = cell(ntrls, 1);
for itrl = 1:ntrls

    select_cfg = [];
    select_cfg.trials = itrl;
    tmp_out = ft_selectdata(select_cfg, dat);

    frctl = ft_freqanalysis(cfg, tmp_out);
    
    cllct_fooof = {frctl.fooofparams(:).aperiodic_params};
    cllct_fooof = cat(1, cllct_fooof{:});
    feats_mat{itrl} = cllct_fooof;

end

toc

feats_mat = cat(3, feats_mat{:});

nparcs = size(feats_mat, 1);
sngl_parc_feats = cell(nparcs, 1);

for iparc = 1:nparcs

    sngl_parc_feats{iparc} = squeeze(feats_mat(iparc, :, :))';

end


%% cat data into the original F structure
% ideally, this could be implemented in another mv function

% single parcels (or chans) 
if isempty(F.single_parcels)
    % create a subfield of the dat structure storing features 
    F.single_parcels = sngl_parc_feats;
else
    for iparc = 1:length(F.single_parcels)
        F.single_parcels{iparc} = [F.single_parcels{iparc}, sngl_parc_feats{iparc}];
    end
end

% all feats for all parcels, concatenated in one mat
F.single_feats.fooof_offset = squeeze(feats_mat(:, 1, :))';
F.single_feats.fooof_slope = squeeze(feats_mat(:, 2, :))';


% log runtime
F.runtime.fooof_aperiodic = round(toc, 2);


end



%% select parcels based on specific constraints
% defined 
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '../STRG_decoding_accuracy/';
helper_functions_path   = '../helper_functions/';
resources_path          = '../../Resources';


addpath(helper_functions_path); addpath(resources_path)
addpath(fieldtrip_path); 
ft_defaults


% output folder
in_feats_path = '../STRG_decoding_accuracy';

text_prompt = 'visual';

% get the logical mask for the parcels containing the text prompt
mask_parcel = mv_select_parcels(text_prompt);

nsubjs = 29; 

[whit_mat, nonwhit_mat] = deal(nan(sum(mask_parcel), nsubjs));

for isubj = 1:nsubjs

    subjcode = sprintf('%0.2d', isubj);
    fname = [subjcode '_WHIT_vs_NOWHIT_parcels.csv'];

    prcls = readtable(fullfile(in_feats_path, fname), "VariableNamingRule","preserve");

    whit_mat(:, isubj) = prcls.decode_accuracy_WHIT;
    nonwhit_mat(:, isubj) = prcls.decode_accuracy_NOWHIT;

end


%% plot

% prepare atlases
atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
[dat_whit, dat_nonwhit] = deal(zeros(1,64984));
filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});

avg_acc_WHIT = mean(whit_mat, 2);
avg_acc_NOWHIT = mean(nonwhit_mat, 2);

acc_index = 0; oridxs = find(mask_parcel)';
for ilab=oridxs
    acc_index=acc_index+1;
    tmp_roiidx=find(atlas.indexmax==ilab);   
    dat_whit(tmp_roiidx)=avg_acc_WHIT(acc_index);
    dat_nonwhit(tmp_roiidx)=avg_acc_NOWHIT(acc_index);
end

this_clims = [.5, .9];

figure(); atlas.data = dat_whit;
plot_hcp_surfaces(atlas,sourcemodel,'Purples',0, ...
                  'accuracy',[-90,0],[90,0],'whitened', this_clims);


figure(); atlas.data = dat_nonwhit;
plot_hcp_surfaces(atlas,sourcemodel,'Greens',0, ...
                  'accuracy',[-90,0],[90,0],'non whitened', this_clims);


% figure(); atlas.data = dat_whit-dat_nonwhit;
% plot_hcp_surfaces(atlas,sourcemodel,'-RdBu',0, ...
%                   'accuracy',[-90,0],[90,0],'whitened vs non-whitened', [-.01, .01]);


%% select the area more consistently away from chance
% so z vals against .5 rather than simple average accuracy. This to account
% for subjects dispersion

zvals_whit = (avg_acc_WHIT-.5)./std(whit_mat, [], 2);
zvals_nonwhit = (avg_acc_NOWHIT-.5)./std(nonwhit_mat, [], 2);

[val1, idx1] = max(zvals_whit);
[val2, idx2] = max(zvals_nonwhit);

[val3, idx3] = max(avg_acc_WHIT);
[val4, idx4] = max(avg_acc_NOWHIT);


% reconvert iindexes from the reduced to the original
best_oridxs = oridxs([idx1, idx2, idx3, idx4]);

%% check the original spectra in the EC/EO conditions for the best parcel(s)

data_path = '/remotedata/AgGross/Fasting/NC/resultsNC/resting_state/source/lcmv';


for isubj = 1:nsubjs


    subjcode = sprintf('%0.2d', isubj);
    fname_in = ['S' subjcode '_control_hcpsource_1snolap.mat'];
    
    % load sample dataset
    temp = load(fullfile(data_path, fname_in));
    dat = temp.sourcedata;
    
    %% merge datasets & set config
    % unitary label
    Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';
    % merge
    cfg = [];
    dat = ft_appenddata(cfg, dat{1}, dat{2});
    % call for config
    cfg_feats = mv_features_cfg();
    
    % select only the best parcels
    cfg = [];
    cfg.channel = dat.label(best_oridxs);
    dat = ft_preprocessing(cfg, dat);

    %% scale up data
    
    dat.trial = cellfun(@(x) x*1e11, dat.trial, 'UniformOutput',false);
    
    %% compute derivative
    
    whitened_dat = dat;
    whitened_dat.trial = cellfun(@(x) diff(x,1,2), dat.trial, 'UniformOutput',false);
    whitened_dat.time = cellfun(@(x) x(2:end), dat.time, 'UniformOutput',false);
    mat_tinfo_adapt = [0:length(dat.trial)-1; 1:length(dat.trial)]';
    whitened_dat.sampleinfo = whitened_dat.sampleinfo-mat_tinfo_adapt;

    %% compute FFT

    out_FFT = ft_freqanalysis(cfg_feats.cfg_FFT, dat);
    out_FFT_whitened = ft_freqanalysis(cfg_feats.cfg_FFT, whitened_dat);

    %% normalization (dB) an
    eceo_WHIT_db = 10*log10(squeeze(mean(out_FFT_whitened.powspctrm, 2)));
    eceo_noWHIT_db = 10*log10(squeeze(mean(out_FFT.powspctrm, 2)));

    EC_WHIT_db = mean(eceo_WHIT_db(Y==1, :));
    EO_WHIT_db = mean(eceo_WHIT_db(Y==2, :));

    EC_noWHIT_db = mean(eceo_noWHIT_db(Y==1, :));
    EO_noWHIT_db = mean(eceo_noWHIT_db(Y==2, :));

    %% merge matrices together

    if isubj == 1
        [WHIT_FFT, noWHIT_FFT] = deal(nan(2, length(EC_WHIT_db), nsubjs));
    end

    WHIT_FFT(1, :, isubj) = EC_WHIT_db;
    WHIT_FFT(2, :, isubj) = EO_WHIT_db;
    
    noWHIT_FFT(1, :, isubj) = EC_noWHIT_db;
    noWHIT_FFT(2, :, isubj) = EO_noWHIT_db;

end

%% plots FFT

figure()

subplot(2, 2, 1); hold on

x_ = out_FFT.freq; 
avg_FFT_nowhit = mean(noWHIT_FFT, 3)';
stderr_FFT_nowhit = std(noWHIT_FFT,[], 3)'./sqrt(nsubjs);

diffs_noWHIT = squeeze(noWHIT_FFT(1, :, :) - noWHIT_FFT(2, :, :));
tvals_noWHIT = sqrt(nsubjs)*mean(diffs_noWHIT, 2)./std(diffs_noWHIT,[], 2);

shadedErrorBar(x_, avg_FFT_nowhit(:, 1), stderr_FFT_nowhit(:, 1), ...
               'lineProps', 'b')

shadedErrorBar(x_, avg_FFT_nowhit(:, 2), stderr_FFT_nowhit(:, 2), ...
               'lineProps', 'r')

legend('EC', 'EO')

title('FFT, no whitening', 'FontSize',13)

subplot(2, 2, 3); hold on

plot(x_, tvals_noWHIT, 'k', 'LineWidth',3)
thresh_sig = tinv(.975, nsubjs-1);
plot(x_, ones(length(tvals_noWHIT), 1)*thresh_sig, '--r', 'LineWidth',1)
legend('t-values', 'sig threshold')
ylim([0, 7])

title({'FFT, no whitening', 't-values EC-EO'}, 'FontSize',13)



subplot(2, 2, 2); hold on

x_ = out_FFT.freq; 
avg_FFT_whit = mean(WHIT_FFT, 3)';
stderr_FFT_whit = std(WHIT_FFT,[], 3)'./sqrt(nsubjs);

diffs_WHIT = squeeze(WHIT_FFT(1, :, :) - WHIT_FFT(2, :, :));
tvals_WHIT = sqrt(nsubjs)*mean(diffs_WHIT, 2)./std(diffs_WHIT,[], 2);

shadedErrorBar(x_, avg_FFT_whit(:, 1), stderr_FFT_whit(:, 1), ...
               'lineProps', 'b')

shadedErrorBar(x_, avg_FFT_whit(:, 2), stderr_FFT_whit(:, 2), ...
               'lineProps', 'r')

title('FFT, pre-whitened', 'FontSize',13)

subplot(2, 2, 4); hold on

plot(x_, tvals_WHIT, 'k', 'LineWidth',3)
thresh_sig = tinv(.975, nsubjs-1);
plot(x_, ones(length(tvals_WHIT), 1)*thresh_sig, '--r', 'LineWidth',1)
ylim([0, 7])

title({'FFT, pre-whitened', 't-values EC-EO'}, 'FontSize',13)



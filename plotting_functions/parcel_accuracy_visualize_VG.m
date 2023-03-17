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

VG_mat = nan(sum(mask_parcel), nsubjs);

for isubj = 1:nsubjs

    subjcode = sprintf('%0.2d', isubj);
    fname = [subjcode '_VG_accs_parcels.csv'];

    prcls = readtable(fullfile(in_feats_path, fname), "VariableNamingRule","preserve");

    VG_mat(:, isubj) = prcls.decode_accuracy_VG;

end


%% plot

% prepare atlases
atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
[dat_VG, dat_nonwhit] = deal(zeros(1,64984));
filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});

avg_acc_VG = mean(VG_mat, 2);

acc_index = 0; oridxs = find(mask_parcel)';
for ilab=oridxs
    acc_index=acc_index+1;
    tmp_roiidx=find(atlas.indexmax==ilab);   
    dat_VG(tmp_roiidx)=avg_acc_VG(acc_index);
end

this_clims = [.7, .9];

figure(); atlas.data = dat_VG;
plot_hcp_surfaces(atlas,sourcemodel,'Purples',0, ...
                  'accuracy',[0,0],[0,0],{'Visual Gamma response', ...
                  'Decoding Accuracy'}, this_clims);



%% select the area more consistently away from chance
% so z vals against .5 rather than simple average accuracy. This to account
% for subjects dispersion

zvals_whit = (avg_acc_VG-.5)./std(VG_mat, [], 2);

[val1, idx1] = max(zvals_whit);

[val3, idx3] = max(avg_acc_VG);


% reconvert iindexes from the reduced to the original
best_oridxs = oridxs([idx1, idx3]);

%% check the original spectra in the EC/EO conditions for the best parcel(s)

data_path = '/remotedata/AgGross/Fasting/NC/resultsNC/visual_gamma/source';


for isubj = 1:nsubjs


    subjcode = sprintf('%0.2d', isubj);
    fname_in = ['S' subjcode '_satted_source_VG.mat'];
    
    % load sample dataset
    temp = load(fullfile(data_path, fname_in));
    sourcedata = temp.sourcedata;

    % redefine trials for pre and post stim segments
    cfg_pre = [];
    cfg_pre.toilim = [-1, 0];
    dat_pre = ft_redefinetrial(cfg_pre, sourcedata);
    
    cfg_stim = [];
    cfg_stim.toilim = [1, 2];
    dat_stim = ft_redefinetrial(cfg_stim, sourcedata);
    
    % merge the data together in a format that allow backward compatibility
    % with EC EO 
    dat = {dat_pre, dat_stim};
    
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
    
    %% compute FFT

    out_FFT = ft_freqanalysis(cfg_feats.cfg_FFT, dat);

    %% normalization (dB) an
    VG_db_mat = 10*log10(squeeze(mean(out_FFT.powspctrm, 2)));

    VG_db = mean(VG_db_mat(Y==1, :));
    prestim_db = mean(VG_db_mat(Y==2, :));

    %% merge matrices together

    if isubj == 1
        VG_FFT = deal(nan(2, length(VG_db), nsubjs));
    end
    
    VG_FFT(1, :, isubj) = VG_db;
    VG_FFT(2, :, isubj) = prestim_db;

end

%% plots FFT

figure()

subplot(2, 1, 1); hold on

x_ = out_FFT.freq; 
avg_FFT_VG = mean(VG_FFT, 3)';
stderr_FFT_VG = std(VG_FFT,[], 3)'./sqrt(nsubjs);

diffs_VG = squeeze(VG_FFT(1, :, :) - VG_FFT(2, :, :));
tvals_VG = sqrt(nsubjs)*mean(diffs_VG, 2)./std(diffs_VG,[], 2);

shadedErrorBar(x_, avg_FFT_VG(:, 1), stderr_FFT_VG(:, 1), ...
               'lineProps', 'b')

shadedErrorBar(x_, avg_FFT_VG(:, 2), stderr_FFT_VG(:, 2), ...
               'lineProps', 'r')

legend('baseline', 'visual stimulation')
xlim([1, 100])

title('FFT, baseline vs visual stimulation', 'FontSize',13)

subplot(2, 1, 2); hold on

plot(x_, tvals_VG, 'k', 'LineWidth',3)
thresh_sig = tinv(.975, nsubjs-1);
plot(x_, ones(length(tvals_VG), 1)*thresh_sig, '--r', 'LineWidth',1)
thresh_sig = tinv(.025, nsubjs-1);
plot(x_, ones(length(tvals_VG), 1)*thresh_sig, '--r', 'LineWidth',1)

legend('t-values', 'sig threshold')
xlim([1, 100])

title({'FFT, no whitening', 't-values baseline-stimulation'}, 'FontSize',13)


%% EC/EO decoding
clearvars;
close all
clc

%% path definition

software_path           = '../../Software/biosig-master/biosig/t300_FeatureExtraction';
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '../DATA/';
entropy_functions_path  = '../../Software/mMSE-master';
helper_functions_path   = '../helper_functions/';
beta_functions_path     = '../beta_functions';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';


addpath(software_path); addpath(helper_functions_path)
addpath(beta_functions_path); addpath(entropy_functions_path)
addpath(plotting_functions_path); addpath(resources_path)
addpath(fieldtrip_path); 
ft_defaults

% load sample dataset
load(fullfile(data_path, 'dat_ECEO.mat'))

%% merge datasets & set config
% unitary label
Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';
% merge
cfg = [];
dat = ft_appenddata(cfg, dat{1}, dat{2});
% call for config
cfg_feats = mv_features_cfg();

%% initialize Feature structure
ntrials = length(dat.trial);

F.single_parcels = [];
F.single_feats = [];
F.multi_feats = double.empty(ntrials, 0);

%% compute frequency features

F = mv_features_freqdomain(cfg_feats, dat, F);

%% compute time features

F = mv_features_timedomain(cfg_feats, dat, F);

%% periodic & aperiodic

F = mv_periodic_aperiodic(cfg_feats, dat, F);

%% classifiers table comparison

tbl_classifiers = mv_features_compare(cfg_feats, F, Y);

%% spatial dist classification accuracy

parc_acc = mv_classify_parcels(cfg_feats, F, Y);

%% plots

% plot accuracy over parcellated brain
atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
atlas.data = zeros(1,64984);
filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});

for ilab=1:length(atlas.indexmaxlabel)
    tmp_roiidx=find(atlas.indexmax==ilab);   
    atlas.data(tmp_roiidx)=parc_acc.accuracy(ilab);
end

figure()
plot_hcp_surfaces(atlas,sourcemodel,'YlOrRd',0, ...
                  'accuracy',[-90,0],[90,0],'SVM accuracy', [.5, 1]);


%% continue from here >

% c = cvpartition(Y,"Holdout",0.2);
% trainIdx=training(c);
% testIdx=test(c);
% acc=zeros(1,360);
% 
% % you can also do it on prewhitened data
% % cfg=[];
% % cfg.derivative='yes'; %prewhitening 
% % for k=1:2,
% %     dat{k}=ft_preprocessing(cfg,dat{k});
% % end
% 
% for k=1:2,
%     Xt{k}=compute_features_time(dat{k});
%     Xf{k}=compute_features_freq(dat{k});
% end
% 
% parfor k=1:360,
%     X=zeros(217,17);
%     X(1:nt{1},1:8)=squeeze(Xt{1}(:,k,:));
%     X(nt{1}+1:end,1:8)=squeeze(Xt{2}(:,k,:));
%     X(1:nt{1},9:17)=squeeze(Xf{1}(:,k,:));
%     X(nt{1}+1:end,9:17)=squeeze(Xf{2}(:,k,:));
%   
%     %X=[squeeze(X1(:,k,:)); squeeze(X2(:,k,:))];
%     X=zscore(X,1,1);
% 
%     %Mdl = fitcknn(X(trainIdx,:),Y(trainIdx),'Standardize',false,'OptimizeHyperparameters','auto',...
%     %'HyperparameterOptimizationOptions',...
%     %struct('AcquisitionFunctionName','expected-improvement-plus','UseParallel',true,'ShowPlots',false,'Verbose',0))
% %     Mdl = fitcdiscr(X(trainIdx,:),Y(trainIdx),'OptimizeHyperparameters','auto',...
% %     'HyperparameterOptimizationOptions',...
% %     struct('AcquisitionFunctionName','expected-improvement-plus','UseParallel',true,'ShowPlots',false,'Verbose',0));
% 
%     %Mdl = fitcknn(X,Y,'NumNeighbors',3,'KFold',10);
%     %[Mdl] = fitcsvm(X,Y,'KernelFunction','gaussian','KernelScale',3.7,'BoxConstraint',1,'KFold',10);
%     Mdl = fitcdiscr(X,Y,'KFold',10);
%     acc(1,k)=1-kfoldLoss(Mdl);
%     %acc(1,k) = 1 - loss(Mdl,X(testIdx,:),Y(testIdx));
%     %cvmodel = crossval(Mdl,'KFold',10);
%     %acc(2,k)=1-kfoldLoss(cvmodel);
% end
% 
% [idx,scores] = relieff(X,Y,3,'method','classification');
% [idx,scores] = fscchi2(X,Y);
% [idx,scores] = fscmrmr(X,Y);
% mm = fscnca(X,Y);
% %mostly entropy and std and wpH
% 
% 
% %% hctsa
%  %featVector = TS_CalculateFeatureVector(dat{1}.trial{1}(1,:)',false);
% 
% cfg=[];
% cfg.channel=1;
% dat1=ft_selectdata(cfg,dat{1});
% dat2=ft_selectdata(cfg,dat{2});
% timeSeriesData=ft_appenddata([],dat1,dat2);
% timeSeriesData=timeSeriesData.trial;
% keywords=[];
% m=1; for k=1:length(dat{1}.trial),keywords{m}='ec';labels{m}=num2str(m);m=m+1;end;
% for k=1:length(dat{2}.trial),keywords{m}='eo';labels{m}=num2str(m);m=m+1;end;
% 
%  save('ECEO_roi1','timeSeriesData','labels','keywords')
%  %transfer to cluster
% TS_Init('ECEO_roi1','hctsa',false)
% TS_Compute(true)  %uses parpool; probably better to run trials in parallel
% 
% TS_InspectQuality('summary');
% TS_LabelGroups('raw',{});
% 
% TS_Normalize('mixedSigmoid',[0.8,1.0]);
% 
% TS_PlotLowDim('norm','pca');
% 
% TS_Classify('norm')
% 
% TS_CompareFeatureSets()
% TS_ClassifyLowDim()
% 
% TS_TopFeatures()
% 
% 



% %% freq
% 
% cfg=[];
% cfg.method='mtmfft';
% cfg.taper='dpss';
% cfg.output='pow';
% cfg.keeptrials='yes';
% cfg.tapsmofrq=1;
% cfg.foilim=[0.25 45];%avoid 0 for 1/f fit
% freq=ft_freqanalysis(cfg,dat);
% 
% 
% %% descriptives
% 
% % important note 1
% % below 1 Hz, the spectrum did not seem linear (after the log log
% % transform). For better linear fit, I include freqs from 1 Hz on.
% maskAbove1Hz = freq.freq>=1;
% chan = 17; % R IPS, out of sympathy
% 
% OneCortexParc = squeeze(freq.powspctrm(:, chan, :));
% avg_spctr = mean(OneCortexParc);
% 
% randtrl = squeeze(OneCortexParc(randi(217), :));
% 
% figure; 
% subplot(1, 2, 1, 'XScale', 'log', 'YScale', 'log'); hold on;
% loglog(freq.freq, avg_spctr, 'LineWidth',2)
% loglog(freq.freq(maskAbove1Hz), avg_spctr(maskAbove1Hz), 'LineWidth',2)
% title('Average (R IPS)', 'FontSize',13)
% 
% subplot(1, 2, 2, 'XScale', 'log', 'YScale', 'log'); hold on;
% loglog(freq.freq, randtrl, 'LineWidth',2)
% loglog(freq.freq(maskAbove1Hz), randtrl(maskAbove1Hz), 'LineWidth',2)
% title('Single trials (R IPS)', 'FontSize',13)
% 
% 
% %%
% freqmap = freq.freq(1:end-2);
% spctrmap = logspctr(1:end-2);
% dx1_spectra = diff(logspctr); dx1_spectra = dx1_spectra(1:end-1);
% dx2_spectra = diff(logspctr, 2);
% 
% %% crit 1:
% 
% krnlpeak = linspace(-1, 1, 5); % sin(linspace(-pi, pi, 5)); %
% test = conv(dx1_spectra, krnlpeak, 'same');
% 
% 
% 
% map = test>0;
% 
% Krnl = gausswin(5);
% mask = conv(map, Krnl, 'same')>=(sum(Krnl));
% 
% figure; hold on;
% plot(freqmap, test)
% plot(freqmap, mask)
% 
% %% fooof
% 
% tic
% cfg               = [];
% cfg.foilim        = [1 45];
% cfg.pad           = 4;
% cfg.tapsmofrq     = 2;
% cfg.method        = 'mtmfft';
% cfg.output        = 'fooof_aperiodic';
% fractal = ft_freqanalysis(cfg, dat);
% toc
% 
% 
% %%
% 


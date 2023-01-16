%% EC/EO decoding
clearvars;
close all
clc

%% path definition

software_path = '../../Software/biosig-master/biosig/t300_FeatureExtraction';
fieldtrip_path = '~/toolboxes/fieldtrip-20221223';
data_path = '../DATA/';
entropy_functions_path = '../../Software/mMSE-master';
helper_functions = '../helper_functions/';

addpath(software_path)
addpath(helper_functions)
addpath(entropy_functions_path)
addpath(fieldtrip_path)
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

%% compute features

dat = mv_features_timedomain(cfg_feats, dat);

%% first test

Mdl = fitcsvm(dat.F.sampen,Y,'KernelFunction','gaussian','KernelScale',...
              3.7,'BoxConstraint',1,'KFold',10);
acc =1-kfoldLoss(Mdl);

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

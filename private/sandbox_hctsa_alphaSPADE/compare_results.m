clearvars
close all
clc

sourcedir = '/remotedata/AgGross/TBraiC/AlphaSpade/';
d=dir([sourcedir, '/*bacc_clinear*']);

load(fullfile('/remotedata/AgGross/TBraiC/AlphaSpade', 'OP.mat'))
feat_codes = OP.CodeString;

full_bacc = nan(length(d), length(feat_codes));

for k0=1:length(d)

    tmp_tr = load(fullfile(d(k0).folder, d(k0).name));
    full_bacc(k0, :) = sort(tmp_tr.bacc, 'descend');

end


%%

full_bacc(full_bacc==0)= nan;
avg_bacc = mean(full_bacc, 1);

srtd_ = sort(avg_bacc, 'descend');
srtd_(isnan(srtd_)) = [];

figure()
plot(srtd_)

%%

tmp_infold = '/remotedata/AgGross/TBraiC/MV-eye/STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim_linearclass_extended';
tbl = readtable(fullfile(tmp_infold, '1331.csv'));
% 
% tbl2 = readtable(fullfile(tmp_infold, '2023.csv'));
% 


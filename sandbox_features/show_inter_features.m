%% EC/EO decoding
clearvars;
% close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
helper_functions_path   = '../helper_functions/';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '../../Software/catch22/wrap_Matlab';


in_feat_path          = '../STRG_computed_features';

addpath(helper_functions_path)
addpath(plotting_functions_path); addpath(resources_path)


%% loop into subjects

nsubjs = 22; multi_subj = [];

for isubj = 1:nsubjs

    subjcode = sprintf('%0.2d', isubj);
    fname_in = [subjcode '_feats.mat'];
    
    % load sample dataset
    temp = load(fullfile(in_feat_path, fname_in));
    F = temp.variableName;

    feat_names = fieldnames(F.single_feats)';

    for this_feat = feat_names

        this_feat_name = this_feat{1};

        if ~strcmp(this_feat_name, 'covFFT') && ~strcmp(this_feat_name , 'fooof_aperiodic')

            temp = F.single_feats.(this_feat_name );
%             temp = normalize(temp, 'scale', 'iqr');
            temp = normalize(temp);
            
            if isubj == 1

                multi_subj.(this_feat_name) = temp;

            else

                multi_subj.(this_feat_name) = cat(1, multi_subj.(this_feat_name), temp);

            end

        end

    end

    if isubj == 1; Y = F.Y;
    else; Y = [Y, F.Y]; end

end


%% compute features similarity

feat_names = fieldnames(multi_subj)'; nfeats = length(feat_names);
mat_corrs = nan(nfeats);

for icol = 1:nfeats

    temp1 = multi_subj.(feat_names{icol});

    if icol == 1
        nparc = size(temp1, 2);
        [kmetrics, ks_pvals] = deal(nan(nfeats));
    end

    for iparc = 1:nparc

       this_parc = temp1(:, iparc);
       vct_cond1 = this_parc(Y==1); vct_cond2 = this_parc(Y==2);
       [~, p, k] = kstest2(vct_cond1, vct_cond2);

       kmetrics(icol, iparc) = k; ks_pvals(icol, iparc) = p;

    end

    for irow = 1:nfeats

        temp2 = multi_subj.(feat_names{irow});
        mat_corrs(irow, icol) = corr2(temp1, temp2);

    end


    fprintf([feat_names{icol}, '\n'])

end

%% 

temp_mat = mat_corrs; 

isupper = logical(triu(ones(size(temp_mat)),1));
temp_mat(isupper) = NaN;

tmp = brewermap(1000, 'RdYlBu'); tmp = flip(tmp);

frmttd_feat_names = cellfun(@(x) strrep(x, '_', '\_'), feat_names, ...
                            'UniformOutput',false);
figure; 
h = heatmap(temp_mat, 'Colormap',tmp, 'MissingDataColor','w'); clim([-1, 1])
h.GridVisible = 'off';
h.YData = frmttd_feat_names;
h.XData = frmttd_feat_names;
title('correlation across features')

%% ks pars

spatial_k_metric = kmetrics*kmetrics';
spatial_k_metric(isupper) = NaN;

tmp = brewermap(1000, 'Reds'); 

figure()
h = heatmap(spatial_k_metric, 'Colormap',tmp, 'MissingDataColor','w');
h.GridVisible = 'off';
h.YData = frmttd_feat_names;
h.XData = frmttd_feat_names;
title('KS metrics (across parcels)')
% clim([0, 4])


%%

figure()
h = heatmap(kmetrics, 'Colormap',tmp, 'MissingDataColor','w');
h.GridVisible = 'off';
h.YData = frmttd_feat_names;
% h.XData = [];
title('KS metrics x parcels')


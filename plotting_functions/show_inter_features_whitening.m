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


data_path               = '/remotedata/AgGross/Fasting/NC/resultsNC/resting_state/source/lcmv';

in_feat_path          = '../STRG_computed_features';

addpath(helper_functions_path)
addpath(plotting_functions_path); addpath(resources_path)


%% loop into subjects

nsubjs = 29; multi_subj = [];

for isubj = 1:nsubjs

    subjcode = sprintf('%0.2d', isubj);
    fname_in_whit = [subjcode '_WHIT_feats.mat'];
    fname_in_nonwhit = [subjcode '_NOWHIT_feats.mat'];

    % load sample dataset
    temp_whit = load(fullfile(in_feat_path, fname_in_whit));
    F_whit = temp_whit.variableName;

    % load sample dataset
    temp_NONwhit = load(fullfile(in_feat_path, fname_in_nonwhit));
    F_NONwhit = temp_NONwhit.variableName;

    feat_names = fieldnames(F_whit.single_feats)';

    % merge whit and nonwhit (for synergistic decoding)
    F_syn = F_whit;
    
    for iparc = 1:length(F_NONwhit.single_parcels)
        F_syn.single_parcels{iparc} = [F_syn.single_parcels{iparc}, ...
                                       F_NONwhit.single_parcels{iparc}];
    end


    for ifeat = 1:length(feat_names)

        this_feat = feat_names{ifeat};
        F_syn.single_feats.(this_feat) = [F_syn.single_feats.(this_feat), ...
                                          F_NONwhit.single_feats.(this_feat)];  

    end
    
    % store
    saveinparfor(fullfile(in_feat_path, [subjcode '_SYN_feats.mat']), F_syn)

    % fetch label
    Y = F_NONwhit.Y; % common to both whit and nonwhit

    % get number of parcels for preallocation
    nparc = length(F_whit.single_parcels);


    if isubj == 1

        mat_corr2 = nan(length(feat_names), nsubjs);
        mat_ks_diffs = nan(length(feat_names), nparc, nsubjs);

    end

    
    for ifeat = 1:length(feat_names)

        this_feat_name = feat_names{ifeat}; 
        temp_whit = F_whit.single_feats.(this_feat_name );
        temp_NONwhit = F_NONwhit.single_feats.(this_feat_name );
            
        % compute corr2
        r = corr2(temp_whit, temp_NONwhit);

        % compute ks metric
        [ks_whit, ks_NONwhit] = deal(nan(nparc, 1));

        for iparc = 1:nparc

            this_parc_whit = temp_whit(:, iparc);
            vct_cond1_whit = this_parc_whit(Y==1); vct_cond2_whit = this_parc_whit(Y==2);

            this_parc_NONwhit = temp_NONwhit(:, iparc);
            vct_cond1_NONwhit = this_parc_NONwhit(Y==1); vct_cond2_NONwhit = this_parc_NONwhit(Y==2);

            try
                [~, p, k_whit] = kstest2(vct_cond1_whit, vct_cond2_whit);
            catch
                p=nan; k_whit=nan;
            end

            try
                [~, p, k_NONwhit] = kstest2(vct_cond1_NONwhit, vct_cond2_NONwhit);
            catch
                p_NON=nan; k_NONwhit=nan;
            end

            ks_whit(iparc) = k_whit;
            ks_NONwhit(iparc) = k_NONwhit;

        end

        % store outputs
        mat_corr2(ifeat, isubj) = r;
        mat_ks_diffs(ifeat, :, isubj) = ks_whit-ks_NONwhit;

    end


end

%% 

frmttd_feat_names = cellfun(@(x) strrep(x, '_', '\_'), feat_names, ...
                             'UniformOutput',false);

avg_r = mean(mat_corr2, 2); err_r = std(mat_corr2, [], 2)/sqrt(nsubjs);
figure()
errorbar(1:length(feat_names), avg_r, err_r, 'vertical', '.')
set(gca,'Xtick', 1:length(feat_names),'XtickLabel',frmttd_feat_names);

h=gca; h.XAxis.TickLength = [0 0];

view([90 -90])

title('Correlation whitened/non whitened', 'FontSize',14)
ylabel('average r', 'FontSize', 12)


%% plot heatmap of ks values for featuresXparcels

text_prompt = 'visual';

% get the logical mask for the parcels containing the text prompt
mask_parcel = mv_select_parcels(text_prompt);
% load one original dataset to fetch the parcrels names
fname_in = ['S' subjcode '_control_hcpsource_1snolap.mat'];

% load sample dataset
temp = load(fullfile(data_path, fname_in));
dat = temp.sourcedata;

% get the parcel names
prcl_names = dat{1}.label(mask_parcel);
frmttd_prcl_names = cellfun(@(x) strrep(x, '_ROI', ''), prcl_names, ...
                             'UniformOutput',false);
frmttd_prcl_names = cellfun(@(x) strrep(x, '_', '\_'), frmttd_prcl_names, ...
                             'UniformOutput',false);



avg_ks = squeeze(mean(mat_ks_diffs, 3));

tmp = brewermap(1000, 'RdYlBu'); tmp = flip(tmp);

figure; 
h = heatmap(avg_ks, 'Colormap',tmp, 'MissingDataColor','w'); 
h.GridVisible = 'off';
h.YData = frmttd_feat_names;
h.XData = frmttd_prcl_names;
h.title({'ks diff', 'Whitened- non whitened'})

set(struct(h).NodeChildren(3), 'XTickLabelRotation', 45);

% %% compute features similarity
% 
% feat_names = fieldnames(multi_subj)'; nfeats = length(feat_names);
% mat_corrs = nan(nfeats);
% 
% for icol = 1:nfeats
% 
%     temp1 = multi_subj.(feat_names{icol});
% 
%     if icol == 1
%         nparc = size(temp1, 2);
%         [kmetrics, ks_pvals] = deal(nan(nfeats));
%     end
% 
%     for iparc = 1:nparc
% 
%        this_parc = temp1(:, iparc);
%        vct_cond1 = this_parc(Y==1); vct_cond2 = this_parc(Y==2);
%        try
%            [~, p, k] = kstest2(vct_cond1, vct_cond2);
%        catch
%            p=nan; k=nan;
%        end
%        kmetrics(icol, iparc) = k; ks_pvals(icol, iparc) = p;
% frmttd_feat_names = cellfun(@(x) strrep(x, '_', '\_'), feat_names, ...
%                             'UniformOutput',false);
%     end
% 
%     for irow = 1:nfeats
% 
%         temp2 = multi_subj.(feat_names{irow});
%         mat_corrs(irow, icol) = corr2(temp1, temp2);
% 
%     end
% 
% 
%     fprintf([feat_names{icol}, '\n'])
% 
% end
% 
% %% 
% 
% temp_mat = mat_corrs; 
% 
% isupper = logical(triu(ones(size(temp_mat)),1));
% temp_mat(isupper) = NaN;
% 
% tmp = brewermap(1000, 'RdYlBu'); tmp = flip(tmp);
% 
% frmttd_feat_names = cellfun(@(x) strrep(x, '_', '\_'), feat_names, ...
%                             'UniformOutput',false);
% figure; 
% h = heatmap(temp_mat, 'Colormap',tmp, 'MissingDataColor','w'); clim([-1, 1])
% h.GridVisible = 'off';
% h.YData = frmttd_feat_names;
% h.XData = frmttd_feat_names;
% title('correlation across features')
% 
% %% ks pars
% 
% spatial_k_metric = kmetrics*kmetrics';
% spatial_k_metric(isupper) = NaN;
% 
% tmp = brewermap(1000, 'Reds'); 
% 
% figure()
% h = heatmap(spatial_k_metric, 'Colormap',tmp, 'MissingDataColor','w');
% h.GridVisible = 'off';
% h.YData = frmttd_feat_names;
% h.XData = frmttd_feat_names;
% title('KS metrics (across parcels)')
% % clim([0, 4])
% 
% 
% %%
% 
% figure()
% h = heatmap(kmetrics, 'Colormap',tmp, 'MissingDataColor','w');
% h.GridVisible = 'off';
% h.YData = frmttd_feat_names;
% % h.XData = [];
% title('KS metrics x parcels')
% 

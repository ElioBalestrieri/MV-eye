function wrap_feature_plot()

resources_path        = '../../Resources';
in_feat_path          = '../STRG_computed_features';
fieldtrip_path        = '~/toolboxes/fieldtrip-20221223';

addpath(resources_path)
addpath(fieldtrip_path); ft_defaults; 

filetype = '_high_gamma_55_100_Hz.mat';
feat_name = 'std';
% fooof_par = 'slope';

%% loop into subjects

nsubjs = 22; mat_merged = nan(360, nsubjs);

count_exc = 0;
for isubj = 1:nsubjs

    subjcode = sprintf('%0.2d', isubj);
    fname_in = [subjcode, filetype];
    
    % load sample dataset
    temp = load(fullfile(in_feat_path, fname_in));
    F = temp.variableName;
    
    try

        feat_of_interest = F.single_feats.(feat_name);

    catch

        count_exc = count_exc +1;
        feat_of_interest = nan(size(feat_of_interest)); % under the assumption that the previous subject had this field

    end

    EC_feat = mean(feat_of_interest(F.Y==1, :));
    EO_feat = mean(feat_of_interest(F.Y==2, :));

    if ~strcmp(feat_name, 'fooof_aperiodic')

        mat_merged(:, isubj) = EC_feat-EO_feat;

    else

        temp = EC_feat-EO_feat;
        if strcmp(fooof_par, 'offset')
            mat_merged(:, isubj) = temp(2:2:end);

        elseif strcmp(fooof_par, 'slope')
            mat_merged(:, isubj) = temp(1:2:end);

        end

    end

end
    
tval_sepfeats = sqrt(nsubjs-count_exc) * mean(mat_merged,2, 'omitnan') ./ std(mat_merged, [], 2, 'omitnan');
    
%% plots

% plot accuracy over parcellated brain
atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
atlas.data = zeros(1,64984);
filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});

for ilab=1:length(atlas.indexmaxlabel)
    tmp_roiidx=find(atlas.indexmax==ilab);   
    atlas.data(tmp_roiidx)=tval_sepfeats(ilab);
end

figure()
plot_hcp_surfaces(atlas,sourcemodel,'RdBu',1, ...
                  'tvalues', ...
                  [-90,0],[90,0],[feat_name, ' ', num2str(nsubjs-count_exc), ' subjects'], [-6, 6]);


end


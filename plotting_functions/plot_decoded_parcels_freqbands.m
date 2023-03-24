%% EC/EO decoding
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


% prepare atlases
atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
atlas.data = zeros(1,64984);
filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});


%%

freq_bands_names = {'delta_1_4_Hz', 'theta_4_8_Hz', 'alpha_8_13_Hz', ...
                    'beta_13_30_Hz', 'low_gamma_30_45_Hz', 'high_gamma_55_100_Hz', ...
                    'feats'};

cmaps_combined = {'BuGn', 'YlGn', 'Purples', 'OrRd', 'Blues', 'Oranges', 'Greens'};

for iband = 1:length(freq_bands_names)

    this_band = freq_bands_names{iband};
    fname = [this_band, '_parcels_accs.csv'];
    tbl = readtable(fullfile(data_path, fname));

    parc_acc = table2array(tbl(2, 2:end));

    this_band_atlas = atlas;

    for ilab=1:length(this_band_atlas.indexmaxlabel)
        tmp_roiidx=find(this_band_atlas.indexmax==ilab);   
        this_band_atlas.data(tmp_roiidx)=parc_acc(ilab);
    end

    if strcmp(this_band, 'feats')
        this_band = 'all freqs';
        this_clims = [.4, .9];
    else
        this_clims = [.45, .7];
    end

    this_band = strrep(this_band, '_', '\_');

    figure()
    plot_hcp_surfaces(this_band_atlas,sourcemodel,cmaps_combined{iband},0, ...
                      'accuracy',[-90,0],[90,0],this_band, this_clims);


end



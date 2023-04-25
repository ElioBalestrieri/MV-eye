%% select parcels based on specific constraints
% defined 
clearvars;
close all
clc


%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
helper_functions_path   = '../helper_functions/';
resources_path          = '../../Resources';


addpath(helper_functions_path); 
addpath(resources_path)
addpath(fieldtrip_path); 
ft_defaults


% input folder
in_accs_path = '../STRG_decoding_accuracy';

% output folder 
out_fig_fold = '/home/balestrieri/research_workspace/POSTDOC/FIGS/MV-eye/manuscript/topo_across';

% get the logical mask for the parcels containing the text prompt
text_prompt = 'visual';
mask_parcel = mv_select_parcels(text_prompt);

%% prepare atlas

atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});

%% plot parcellated accuracy for each exp cond

mdls = {'FTM', 'FullFFT', 'FreqBands', 'TimeFeats'};
this_clims = [.5, 1]; % common color limits for all conds

for iCOND = {'ECEO', 'VS'}

    fname = ['AcrossSubjs_' iCOND{1} '__SingleParcels_accuracy.csv'];

    prcls = readtable(fullfile(in_accs_path, fname), "VariableNamingRule","preserve");


    for iMDL = mdls

        vect_acc = prcls.([iMDL{1} '_decoding_accuracy']);
        
        acc_index = 0; oridxs = find(mask_parcel)'; tmp_dat = zeros(1,64984);
        for ilab=oridxs
            acc_index=acc_index+1;
            tmp_roiidx=find(atlas.indexmax==ilab);   
            tmp_dat(tmp_roiidx)=vect_acc(acc_index);
        end
        
        h = figure(); atlas.data = tmp_dat;
        plot_hcp_surfaces(atlas,sourcemodel,'Purples',0, ...
                          'accuracy',[0,20],[0,-20],{[iCOND{1}, ', ' iMDL{1}], ...
                          'Decoding Accuracy'}, this_clims);

        saveas(h, fullfile(out_fig_fold, [iCOND{1}, '_' iMDL{1}, '.png']))

    end


end


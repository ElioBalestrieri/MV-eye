%% select parcels based on specific constraints
% defined 
clearvars;
close all
clc

%% path definition

helper_functions_path = '../helper_functions/';

% output folder
in_feats_path = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features';

text_prompt = 'visual';

% get the logical mask for the parcels containing the text prompt
mask_parcel = mv_select_parcels(text_prompt);

nsubjs = 29; 

for isubj = 1:nsubjs

    subjcode = sprintf('%0.2d', isubj);
    fname_whit = [subjcode '_WHIT_feats.mat'];
    fname_nowhit = [subjcode '_NOWHIT_feats.mat'];

    % load computed features
    temp_whit = load(fullfile(in_feats_path, fname_whit));
    temp_nowhit = load(fullfile(in_feats_path, fname_nowhit));

    F = temp_nowhit.variableName; F_whitened = temp_whit.variableName;

    % apply the mask to 
    F.single_parcels = F.single_parcels(mask_parcel);
    fld_names = fieldnames(F.single_feats); nfields = length(fld_names);
    for ifield = 1:nfields

        this_field = fld_names{ifield};

        try 
            F.single_feats.(this_field) = F.single_feats.(this_field)(:, mask_parcel);
        catch
            sprintf('\nexception reached for %s', this_field)
        end


    end


    % save
    fname_out_feat = [subjcode, '_NOWHIT_feats.mat'];
    fname_out_feat_whit = [subjcode, '_WHIT_feats.mat'];

    saveinparfor(fullfile(out_feat_path, fname_out_feat), F)
    saveinparfor(fullfile(out_feat_path, fname_out_feat_whit), F_whitened)


end

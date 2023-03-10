function mask_selected = mv_select_parcels(text_definer)

% see atlas(es) doc at 
% https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/atlases.html

% atlas table fetched at 
% https://bitbucket.org/dpat/tools/src/master/REF/ATLASES/Glasser_2016_Table.xlsx

HCP_atlas = readtable('HCP-MMP1_UniqueRegionList.csv', 'VariableNamingRule', 'preserve');
cortex_defs = HCP_atlas.cortex;

mask_selected = cellfun(@(x) ~isempty(regexpi(x, text_definer,"once")), ...
    cortex_defs);


end
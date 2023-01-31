function saveinparfor_fbands(pathout, subjcode, vout, Y, cfg_feats)

nbands = length(vout);

for iband =1:nbands

    variableName = vout{iband};

    % append Y labels
    variableName.Y = Y;

    % append cfg for easy check the operations performed before
    variableName.cfg_feats = cfg_feats;

    band_id = variableName.bandIdentifier;
    fname = [subjcode, '_', band_id, '.mat'];

    save(fullfile(pathout, fname), 'variableName')

end

end
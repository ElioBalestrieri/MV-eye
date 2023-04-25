function ThreadLevelFunc(arg, outfold)
% function to be run for each thread, usually a subject


pause(210) % COMMENT OUT, just for testing!!!

% define filename
fname = [sprintf('ID_%0.2d', arg), '.csv'];

% write file
csvwrite(fullfile(outfold, fname), arg)


end

function NodeLevelFunc(arg)
% function example for parallel computation over a node

% output folder
outfold = './STRG_temp/'; if ~isfolder(outfold); mkdir(outfold); end


switch arg

    case 'Job1'

        subjIDs = 1:36;

    case 'Job2'

        subjIDs = 37:72;

        % and so on...

end

% get the number of available threads based on the number
nthreads = length(subjIDs); 
thisObj = parpool(nthreads);

% start parallel for loop
parfor iSubj = subjIDs

    ThreadLevelFunc(iSubj, outfold)

end

% delete parallel object
delete(thisObj)



end

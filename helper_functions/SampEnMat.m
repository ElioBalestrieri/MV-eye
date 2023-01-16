function saen = SampEnMat(dim, r, X)
% matrix implementation of SampEn, allowing to cut computation time by one
% third in a single trial.
%
% Adapted by EB

Nsamples = size(X, 2); Nchans = size(X, 1);
correl = cell(1, 2);
dataMat = zeros(Nchans, Nsamples-dim, dim+1);

for i = 1:dim+1
    dataMat(:, :, i) = X(:, i:Nsamples-dim+i-1);
end

for m = dim:dim+1

    count = zeros(Nchans, Nsamples-dim);
    tempMat = dataMat(:, :, 1:m);

    for i = 1:Nsamples-m

        Xn = tempMat(:, i+1:Nsamples-dim, :);
        Xm = repmat(tempMat(:, i, :), 1, Nsamples-dim-i, 1);
        dist = squeeze(max(abs(Xn-Xm), [], 3));
        D = (dist<r); 

        % squeeze added, differently from the original function, to account
        % for the empty matrix originated at i=end
        if isempty(D); D = zeros(Nchans, 2); end
        
        count(:, i) = sum(D, 2)/(Nsamples-dim);

    end

    correl{m-dim+1} = sum(count, 2)/(Nsamples-dim);

end

saen = log(correl{1}./correl{2});

end
function Hw = wpH(x,n,tau)

% This function calculates the weighted permutation entropy of a data series.
% Details can be found in
% Fadlallah, B., Chen, B., Keil, A., & Príncipe, J. (2013). Weighted-permutation entropy: 
% A complexity measure for time series incorporating amplitude information. 
% Physical Review E, 87(2), 022911. https://doi.org/10.1103/PhysRevE.87.022911
%
% Weighted Permutation entropy is an extension to permutation entropy:
% Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series.
% Physical Review Letters, 88(17), 174102. https://doi.org/10.1103/PhysRevLett.88.174102
% 
% This Code is based on the petropy function by Andreas Müller and the
% related publication:
% Riedl, M.; Müller, A.; Wessel, N.: Practical considerations of 
% permutation entropy. The European Physical Journal Special Topics 
% 222 (2013) 2, 249–262
%
% An application to EEG data in cognitive neuroscience can be found here:
% Waschke, L., Wöstmann, M., & Obleser, J. (2017). States and traits of neural irregularity
% in the age-varying human brain. 
% Scientific Reports, 7(1), 17381. https://doi.org/10.1038/s41598-017-17766-4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% version 1.1, 02.02.2017: Leo Waschke

% x      - data vector (Mx1 or 1xM)
% n      - motif length --> number of possible motifs = n!
           % note that this code has been written and tested for n = 3 only. 
% tau    - time delay factor / time lag
           % should be very small for neurophysiological data, e.g.= 1 

numch = size(x,1);
P = size(x,1);
M = length(x);
x = reshape(x, M,1,P);

shift_mat_ind = repmat(reshape(0:tau:(n-1)*tau,[],1) * ones(1,M-(n-1)*tau) +...
    ones(n, 1) * reshape(1:(M-(n-1)*tau),1,[]),1,1,numch);
addmat = zeros(size(shift_mat_ind));
for c = 1:numch
    addmat(:,:,c) = addmat(:,:,c)+(c-1)*(M-(n-1));
end
shift_mat_ind = shift_mat_ind + addmat;
shift_mat = x(shift_mat_ind);

% allow equal values the same index
ind_mat = zeros(size(shift_mat));
var_mat = zeros(P,length(shift_mat));
for ii=1:size(ind_mat,2)
    [~,ind_mat(:,ii,:)]=sort(squeeze(shift_mat(:,ii,:)),1);
    % take variances of values in snippets
    var_mat(:,ii,:) = var(shift_mat(:,ii,:));
end

for c = 1:numch
    % assign unique number to each pattern (base-n number system)
    ind_vec(c,:) = (n.^(0:n-1)) * (squeeze(ind_mat(:,:,c))-1);
    % extract patterns
    patterns(:,c) = unique(ind_vec(c,:));
    % loop over patterns and compute relative variances for each pattern
end

% calculate variance used for the weighting of pattern probabilities
sumvar = zeros(numch,length(patterns(:,c))); % preallocation
for c = 1:numch
    for k = 1:length(patterns(:,c))
        sumvar(c,k) = sum(var_mat(ind_vec(c,:)==patterns(k,c))./sum(var_mat(c,:)));
    end
end
% do the weighting
Hw = -sum(sumvar .* log2(sumvar),2);
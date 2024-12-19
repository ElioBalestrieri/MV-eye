clear all 
close all
clc


addpath('/remotedata/AgGross/TBraiC/Resources/dPCA/matlab')
load("avg_preproc.mat")

% load fieldnames and adapt them to represent features in space
load('TS-fieldnames.mat')
f{end} = 'fooof_slope'; f = [f; {'fooof_offset'}];

% permute dimensions to have features ad cond indepepndent
perm_dat = permute(dat, [1 4 3 2]);

% combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}};
% margNames = {'Subjects', 'Condition', 'Condition independent', 'Interaction'};
% margColours = [23 100 171; 187 20 25; 150 150 150; 114 97 171]/256;

combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}};
margNames = {'Subjects', 'Condition', 'Cond. ind.'};
margColours = [23 100 171; 187 20 25; 150 150 150]/256;


% perform dPCA
tic
[W,V,whichMarg] = dpca(perm_dat, 15, 'combinedParams', combinedParams);
toc

dpca_perMarginalization(perm_dat, @dpca_plot_default, ...
   'combinedParams', combinedParams);
explVar = dpca_explainedVariance(perm_dat, W, V, 'combinedParams', combinedParams);

[YY, margNums] = dpca_marginalize(perm_dat, ...
                                      'combinedParams', combinedParams);


%% plot expl var as stacked bars

figure() 
subplot(1, 3, 1:2); hold on
b = bar(explVar.margVar', 'stacked');
b(1).FaceColor = margColours(1, :);
b(2).FaceColor = margColours(2, :);
b(3).FaceColor = margColours(3, :);

scatter(1:15, cumsum(sum(explVar.margVar, 1)), 30, [0 0 0], 'filled')
plot(1:15, cumsum(sum(explVar.margVar, 1)), 'k', 'LineWidth', 2)
xlabel('component')
title('Variance explained')

legnames = [margNames, {''}, {'explained var (cumulative sum)'}];
legend(legnames)

subplot(1, 3, 3)
var_accounted = sum(explVar.margVar, 2)/100;
ax = gca();

pie(ax, var_accounted, [1 1 1])
ax.Colormap = margColours; 

%% marginalization 

% margindim
margindim = 2; % 2=conditions; 1=subjects
average_along = 2; % 3=conditions; 2=subjects

% first component
comp1COND = YY{margindim}(1, :, :, :);
avg_comp1COND = squeeze(mean(comp1COND, average_along));

% second component
comp2COND = YY{margindim}(2, :, :, :);
avg_comp2COND = squeeze(mean(comp2COND, average_along));

% the idea: uniform vectors in the complex plane should cancel each other.
% Hence the most widespread a feature, the lowest should be the sum of
% vectors in the complex plane
feats_cmplx_vects = avg_comp1COND + 1i*avg_comp2COND;
dispersion_feats = abs(sum(feats_cmplx_vects));

% % third component
% comp3COND = YY{margindim}(3, :, :, :);
% avgsubj_comp3COND = squeeze(mean(comp3COND, average_along));
% bestFcomp3COND = vecnorm(avgsubj_comp3COND);

% sort and barplot 
[srtd, idxs] = sort(dispersion_feats, 'ascend');
srtd_labels = f(idxs);
frmttd_feat_names = cellfun(@(x) strrep(x, '_', '\_'), ...
    srtd_labels, 'UniformOutput',false);

figure()
barh(srtd)
yticks(1:41)
yticklabels(frmttd_feat_names)

%% marginalization
% rough copy & paste from the plotting function

Xfull = perm_dat;


% centering
X = Xfull(:,:);
X = bsxfun(@minus, X, mean(X,2));
XfullCen = reshape(X, size(Xfull));

% total variance
totalVar = sum(X(:).^2);


[Xmargs, margNums] = dpca_marginalize(XfullCen, ...
                                      'combinedParams', combinedParams, ...
                                      'ifFlat', 'yes');
PCs = [];
vars = [];
margs = [];

ncompsPerMarg = 3;

for m=1:length(Xmargs)
    %[~,S,V] = svd(Xmargs{m});      % this is very slow!
    margVar(m) = sum(Xmargs{m}(:).^2)/totalVar*100;
    
    if size(Xmargs{m},1)<size(Xmargs{m},2)
        XX = Xmargs{m}*Xmargs{m}';
        [U,S] = eig(XX);
        [~,ind] = sort(abs(diag(S)), 'descend');
        S = sqrt(S(ind,ind));
        U = U(:,ind);
        SV = U'*Xmargs{m};
    else
        XX = Xmargs{m}'*Xmargs{m};
        [U,S] = eig(XX);
        [~,ind] = sort(abs(diag(S)), 'descend');
        S = sqrt(S(ind,ind));
        U = U(:,ind);
        SV = (U*S)';
    end        
    
    %PCs = [PCs; S(1:10,1:10)*V(:,1:10)'];
    PCs = [PCs; SV(1:ncompsPerMarg,:)];
    vars = [vars; diag(S(1:ncompsPerMarg,1:ncompsPerMarg)).^2];
    margs = [margs repmat(m, [1 ncompsPerMarg])];
end


[vars,ind] = sort(vars,'descend');
PCs = PCs(ind,:);
margs = margs(ind);
%PCs = PCs(1:15,:);
vars = vars / totalVar * 100;

dims = size(Xfull);
Z = reshape(PCs, [length(ind) dims(2:end)]);

i = 1
cln = {i};
for j=2:ndims(Z)
    cln{j} = ':';
end















figure()
dpca_plot(perm_dat, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'legendSubplot', 16);

%% Step 2: PCA in each marginalization separately

dpca_perMarginalization(perm_dat, @dpca_plot_default, ...
   'combinedParams', combinedParams);

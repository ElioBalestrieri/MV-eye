function class_parcels = mv_classify_parcels(cfg_feats, F, Y)
% classify the full set of features in each parcel

rng default

switch cfg_feats.parcelClass
    
    case 'SVM'
        anon_class_func = @(X_) fitcsvm(X_,Y, 'KFold',cfg_feats.kFoldNum);

    case 'LDA'
        anon_class_func = @(X_) fitcdiscr(X_,Y, 'KFold',cfg_feats.kFoldNum);

    case 'NaiveBayes'
        anon_class_func = @(X_) fitcnb(X_,Y, 'KFold',cfg_feats.kFoldNum);
        
    otherwise
        error('\n"%s" is not a recognized classifier')

end

tic
parcel_Mdls = cellfun(anon_class_func, F.single_parcels, ...
                     'UniformOutput',false);
class_parcels.runtime = toc;
class_parcels.accuracy = cellfun(@(x) 1-kfoldLoss(x), parcel_Mdls);

end


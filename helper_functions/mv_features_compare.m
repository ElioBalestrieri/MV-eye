function acc_feats = mv_features_compare(cfg_feats, F, Y)
% single features (PCA-reduced) accuracy

rng default

% fetch the features computed
features_types = fieldnames(F.single_feats);

% fetch the classifiers to be tested
class_types = cfg_feats.classifiers;

% preallocate table
nClass = numel(class_types); nFeats = numel(features_types);
[tbl_classifiers, tbl_dummy] = deal(array2table(nan(nClass, nFeats), ...
                                    'RowNames',class_types, ...
                                    'VariableNames',features_types'));

% shuffle Y labels for dummy model accuracy
Ydummy = Y(randperm(length(Y)));

% loop cross features and classifiers

icol=0; 
for iFeat = features_types'

    featName = iFeat{1}; icol = icol+1;
    X = F.single_feats.(featName);

    irow=0;
    for iClass = class_types

        className = iClass{1}; irow = irow+1;
    
        if cfg_feats.verbose 
            fprintf('\nFitting %s for %s', className, featName)
        end

        switch className
    
            case 'SVM'
    
                realMdl = fitcsvm(X,Y,'Standardize',true,'KFold',cfg_feats.kFoldNum);
                dummyMdl = fitcsvm(X,Ydummy,'Standardize',true,'KFold',cfg_feats.kFoldNum);
    
            case 'LDA'

                realMdl = fitcdiscr(X,Y,'KFold',cfg_feats.kFoldNum);
                dummyMdl = fitcdiscr(X,Ydummy,'KFold',cfg_feats.kFoldNum);

            case 'NaiveBayes'

                realMdl = fitcnb(X,Y,'KFold',cfg_feats.kFoldNum);
                dummyMdl = fitcnb(X,Ydummy,'KFold',cfg_feats.kFoldNum);
                
            otherwise

                error('\n"%s" is not a recognized classifier')
        
        end

        tbl_classifiers(irow, icol) = {1-kfoldLoss(realMdl)};
        tbl_dummy(irow, icol) = {1-kfoldLoss(dummyMdl)};

    end

end

acc_feats.tbl_realClass = tbl_classifiers;
acc_feats.tbl_dummyClass = tbl_dummy;


end
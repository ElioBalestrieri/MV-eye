function tbl_classifiers = mv_multitest(F, Y)



%% single features (PCA-reduced) accuracy

fn = fieldnames(F.single_feats);


tbl_classifiers = nan(2,2);

col=0;
for ifeat = fn'

    col = col+1;
    X = F.single_feats.(ifeat{1});

    % SVM
    SVM_Mdl = fitcsvm(X,Y, 'KFold',8);
    SVM_acc = 1-kfoldLoss(SVM_Mdl);
    
    tbl_classifiers(1, col) = SVM_acc;

    % LDA
    LDA_Mdl = fitcdiscr(X,Y, 'KFold',8);
    LDA_acc = 1-kfoldLoss(LDA_Mdl);
    
    tbl_classifiers(2, col) = LDA_acc;
    
    ifeat

end





end
function bal_acc = wrap_linear_class(X_tr, X_te, Y_tr, Y_te)

[Mdl] = fitclinear(X_tr,Y_tr);
[Y_pred,scores]=predict(Mdl,X_te);
[c1,c2,c3,c4]=confusion(Y_te',Y_pred');
bal_acc=0.5*(c4(2,3)/(c4(2,3)+c4(2,1)) + c4(2,4)/(c4(2,4)+c4(2,2))); %balanced accuracy


end


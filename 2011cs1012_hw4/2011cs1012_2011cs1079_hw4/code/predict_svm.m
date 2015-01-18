function [pred,acc]=predict_svm(X_test,Y_test,X,Y,alpha,w_0,type)
n=size(X_test,1);
K=zeros(size(X_test,1),size(X,1));
if type==1
    K=X_test*X';
elseif type==2
    K=(1+X_test*X').^2;
end
res=K*(Y.*alpha)+w_0;
pred=(res>0)-1*(res<0);
acc=(sum(pred==Y_test)/n)*100;
end
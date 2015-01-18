function [alpha,w_0]=train_svm(X,Y,C,type)
%calculates alpha and svm offset
%This is a 2-class svm.The label vector Y has labels 1 or -1 for each data
%point.
n=size(X,1);
%Kernel calculation for training
K=zeros(n,n);
if type==1
    K=X*X';
elseif type==2
    K=(1+X*X').^2;
end
    
alpha=mysvmnonseparabledual(X,Y,K,C);
w_0=calc_svm_offset(alpha,Y,K,C);
end
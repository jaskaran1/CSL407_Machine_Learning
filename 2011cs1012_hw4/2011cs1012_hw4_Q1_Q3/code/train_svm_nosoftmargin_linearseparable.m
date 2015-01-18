function [alpha,w_0,w]=train_svm_nosoftmargin_linearseparable(X,Y)
%calculates alpha,svm offset and w
%This is a 2-class svm.The label vector Y has labels 1 or -1 for each data
%point.Y is a column vector.
K=X*X';%Kernel    
d=size(X,2);
alpha=mysvmseparabledual(X,Y,K);%alpha is nx1
w_0=calc_svm_offset_nosoftmargin(alpha,Y,K);
w=sum(((alpha.*Y)*ones(1,d)).*X);%1xd vector
end
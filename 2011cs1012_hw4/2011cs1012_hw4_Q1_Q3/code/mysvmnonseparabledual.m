function alpha=mysvmnonseparabledual(X,Y,K,C)
%find the n dim vector alpha which maximizes the dual problem
%equivalent to saying minimise -(dual problem).Since quadprog minimises
%the given problem
%works only on 2 classes where y_i=1 or -1
%Y is a column vector of nx1
%K is the kernel matrix-nxn
%X->is there any use of X?
n=length(Y);
alpha=zeros(n,1);
beq=zeros(n,1);
f=-ones(n,1);%f is a column vector
H=(Y*Y').*K;
%size(H)
%if H==H'
%    fprintf('Symmetric')
%eig(H)
%    sum(eig(H)<0)    
Aeq=ones(n,1)*Y';
lb=zeros(n,1);
ub=C*ones(n,1);
%opts=optimset('Algorithm',interior-point-convex,'MaxIter',10);%Display iter doesn't work using optimset
%opts=optimset('MaxIter',10);
opts=optimoptions('quadprog','Algorithm','interior-point-convex','MaxIter',1);
[alpha,fval,exitflag,output]=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts);
fval
exitflag
output
end
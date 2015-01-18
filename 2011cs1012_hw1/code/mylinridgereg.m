function W=mylinridgereg(X,T,lambda)
%Trains using linear least squares solution with ridge regression penaly
%parameter lambda
[n,d]=size(X);
X=[ones(n,1) X];%append ones
%W=zeros(d+1,1);
%itermax=100;%Control maximum iterations
%alpha=0.01;%Control learning rate
%fprintf('Lambda=%f\n',lambda);
% for i=1:itermax
%     if i==itermax
%         J=sum((X*W-T).^2)/(2*n)+lambda*sum(W(2:end).^2);
%         fprintf('Cost after %d iteration is %f\n',i,J);
%     end
%     W(1)=W(1)-(alpha/n)*sum((X*W-T).*X(:,1));
%     W(2:end)=W(2:end)-((alpha/n)*((X*W-T)'*X(:,2:end))'+(lambda/n)*W(2:end));
% end
W=pinv(X'*X+lambda*eye(d+1))*X'*T;
end
function [J, grad] = objgradcompute(w, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization
m = length(y); % number of training examples
J = 0;
grad = zeros(size(w));
h=sigmoid(w'*X');%use sigmoid(X*w)
J=1/m*sum(-y'.*log(h)-(1-y').*log(1-h))+lambda/(2*m)*(sum(w.^2)-w(1)^2);%since theta0 isn't included in this
grad=1/m*(h-y')*X+(lambda/m)*w';%since grad has 1xn dim
grad(1)=grad(1)-(lambda/m)*w(1);
end

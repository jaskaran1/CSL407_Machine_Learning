function pred=mylinridgeregeval(X,W)
%predicts values for the dataset using itself and weights
X=[ones(size(X,1),1) X];
pred=X*W;
end
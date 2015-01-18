load('credit.mat');
plotdata(data,label);
hold on;
xlabel('x1');
ylabel('x2');
legend('y=1','y=0','Location','SouthEast');
hold off;
%Transform features
deg=2;
X=featuretransform(data,deg);%The returned X,contains the intercept term
% Initialize parameters between -1 to 1
initial_w=zeros(size(X,2),1);%Currently used zeros for identical results
%Comment this and uncomment the line below for random initialization
%initial_w=-1+(1-(-1)).*rand(size(X,2),1);%w is a column
%Regularization parameter
lambda=0.75;
%Options
options=optimset('GradObj', 'on', 'MaxIter', 500);
%Optimize
[W,J,exit_flag]=fminunc(@(w)(objgradcompute(w,X,label,lambda)),initial_w,options);
%Plot boundary
plotdecisionboundary(W, X,label,deg);
p=predict(W,X);
fprintf('Train Accuracy: %f\n', mean(double(p == label)) * 100);
 title(sprintf('lambda = %g', lambda))
 % Labels and Legend
 xlabel('x1')
 ylabel('x2')
 legend('y = 1', 'y = 0','Decision boundary');
 hold off;
plotdata(data,label);
hold on;
lindiscriminant(X(:,2:end),label,deg);
legend('y = 1', 'y = 0','LDA decision boundary');
  hold off;


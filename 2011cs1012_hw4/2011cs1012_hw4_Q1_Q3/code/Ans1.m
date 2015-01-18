load('dataset1');
s=RandStream.create('mt19937ar','seed',5489);
RandStream.setGlobalStream(s);
n=length(X);
%-------2 categories-labels are -1,1
categories=[-1,1]';
%--------vecsPerCat(i) stores the number of examples with i as their category
vecsPerCat = getVecsPerCat(Y,categories);
numFolds=10;
%---foldSizes is a matrix #categories X #folds
%---foldSize(i,j) is number of examples belonging to 
%---ith category and jth fold
foldSizes = computeFoldSizes(vecsPerCat, numFolds);
%--------Sort them randomly
[X_sorted, y_sorted] = randSortAndGroup(X,Y,categories);

cvaccuracy=0;
s=0;
for Fold=1:numFolds
fprintf('Cross validation round/fold number %d\n',Fold);
% Select the vectors to use for training and cross validation.
[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% posclass=X_train((y_train==1),:);
% negclass=X_train((y_train==-1),:);
% figure;
% plot(posclass(:,1),posclass(:,2),'ko','MarkerFaceColor','r');
% hold on;
% plot(negclass(:,1),negclass(:,2),'ko','MarkerFaceColor','g');
% Train svm using QP
fprintf('Training....\n');
tic;
[alpha,w_0,w]=train_svm_nosoftmargin_linearseparable(X_train,y_train);
fprintf('w is\n');
w
fprintf('w_0 is %f\n',w_0);
t=toc;
s=s+t;
% Test svm on validation set
[pred,acc]=predict_svm(X_val,y_val,X_train,y_train,alpha,w_0,1);
fprintf('y_val\n');
y_val
fprintf('pred\n');
pred
fprintf('Accuracy in fold %d=%f\n',Fold,acc);
cvaccuracy=cvaccuracy+acc;
fprintf('Time taken in fold %d=%f secs\n',Fold,t);
end
s=s/numFolds;
cvaccuracy=cvaccuracy/numFolds;
fprintf('Average Time taken for training=%f\n',s);

%Plot of the hyperplane w'x+b=1
MIN=min(X_train(:,1));
MAX=max(X_train(:,1));
delta=(MAX-MIN)/size(X_train,1);
x=MIN:delta:MAX;
y_pos=(1-w_0-w(1)*x)/w(2);
y_neg=(-1-w_0-w(1)*x)/w(2);
y_zero=(-w_0-w(1)*x)/w(2);
posclass=X_train((y_train==1),:);
negclass=X_train((y_train==-1),:);
plot(posclass(:,1),posclass(:,2),'ko','MarkerFaceColor','r');
hold on;
plot(negclass(:,1),negclass(:,2),'ko','MarkerFaceColor','g');
plot(x,y_pos,'r-',x,y_neg,'g-',x,y_zero,'b-');
title('Plot of margin and decision boundary');
xlabel('Feature1');
ylabel('Feature2');
legend('+ve class','-ve class','+1 margin','decision boundary','-1 margin','Location','South');
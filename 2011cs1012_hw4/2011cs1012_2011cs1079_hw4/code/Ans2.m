load('dataset2');
s=RandStream.create('mt19937ar','seed',5489);
RandStream.setDefaultStream(s);
C=1;
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
type=1;%linear kernel
timeperfold=0;
cvaccuracy=0;

%Using libsvm function with linear kernel we get accuracy 57.45
%svmstruct=svmtrain(Y,X,'-t 0');
%[pred,acc,~]=svmpredict(Y,X,svmstruct);
s=0;
%-------Linear Kernel-----------
for Fold=1:numFolds

fprintf('Cross validation round/fold number %d\n',Fold);
% Select the vectors to use for training and cross validation.
[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% Train svm using QP
fprintf('Training....\n');
tic;
[alpha,w_0]=train_svm(X_train,y_train,C,type);
t=toc;
s=s+t;
% Test svm on validation set
[pred,acc]=predict_svm(X_val,y_val,X_train,y_train,alpha,w_0,type);
fprintf('Accuracy in fold %d=%f\n',Fold,acc);
cvaccuracy=cvaccuracy+acc;
fprintf('Time taken in fold %d=%f secs\n',Fold,t);
end
s=s/numFolds;
fprintf('Average Time taken for training=%f\n',s);
%K is quadratic polynomial kernel
%Using libsvm with poly kernel-98% at >200 iterations
% svmstruct=svmtrain(Y,X,'-t 1 -g 1 -r 1');
% [pred,acc,~]=svmpredict(Y,X,svmstruct);

type=2;%polynomial kernel
cvaccuracy2=0;
t2=0;
%---Polynomial Kernel---------
for Fold=1:numFolds

fprintf('Cross validation round/fold number %d\n',Fold);
% Select the vectors to use for training and cross validation.
[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% Train svm using QP
fprintf('Training....\n');
tic;
[alpha,w_0]=train_svm(X_train,y_train,C,type);
t=toc;
t2=t2+t;
% Test svm on validation set
[pred,acc]=predict_svm(X_val,y_val,X_train,y_train,alpha,w_0,type);
fprintf('Accuracy in fold %d=%f\n',Fold,acc);
cvaccuracy2=cvaccuracy2+acc;

fprintf('Fold %d time = %f\n',Fold,t);
end
t2=t2/numFolds;
fprintf('Time taken per fold for the linear kernel=%f\n',s);
fprintf('Classification Accuracy using linear kernel is %f\n',cvaccuracy/numFolds);
fprintf('Time taken per fold for the polynomial kernel=%f\n',t2);
fprintf('Classification Accuracy using polynomial kernel is %f\n',cvaccuracy2/numFolds);

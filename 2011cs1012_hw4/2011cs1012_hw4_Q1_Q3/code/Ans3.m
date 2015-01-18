load('dataset3');
s=RandStream.create('mt19937ar','seed',5489);
RandStream.setGlobalStream(s);
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

%-------------------------------
%---------------------------------------------IMPLEMENTATIONS START
%--UNCOMMENT THE REQUIRED IMPLEMENTATION
%-------------------------------

%----IMPLEMENTATION 1----
%My SVM implementation(QP) with linear kernel

%----UNCOMMENT FROM THE LINE BELOW TO USE IMPLEMENTATION 1----
% type=1;%linear kernel
% cvaccuracyQPlinear=0;
% timeQPlinear=0;%s stores the cumulative time for each fold
% for Fold=1:numFolds
% fprintf('Cross validation round/fold number %d\n',Fold);
% % Select the vectors to use for training and cross validation.
% [X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% % Train svm using QP
% fprintf('Be Patient!!.Training start...\n');
% tic;
% [alpha,w_0]=train_svm(X_train,y_train,C,type);
% t=toc;
% fprintf('Training finish...\n');
% timeQPlinear=timeQPlinear+t;
% % Test svm on validation set
% [pred,acc]=predict_svm(X_val,y_val,X_train,y_train,alpha,w_0,type);
% fprintf('Accuracy in fold %d=%f\n',Fold,acc);
% cvaccuracyQPlinear=cvaccuracyQPlinear+acc;
% fprintf('Time taken in fold %d=%f secs\n',Fold,t);
% end
% timeQPlinear=timeQPlinear/numFolds;
% cvaccuracyQPlinear=cvaccuracyQPlinear/numFolds;
% fprintf('Time taken per fold for QP linear kernel=%f\n',timeQPlinear);
% fprintf('Classification Accuracy using QP linear kernel is %f\n',cvaccuracyQPlinear);
% return; 
%----UNCOMMENT FROM THE LINE ABOVE TO USE IMPLEMENTATION 1----


%----IMPLEMENTATION 2---
%----Matlab SVM implementation(SMO) with linear Kernel----

%----UNCOMMENT FROM THE LINE BELOW TO USE IMPLEMENTATION 2----
% cvaccuracySMOlinear=0;
% timeSMOlinear=0;%s stores the cumulative time for each fold
% for Fold=1:numFolds
% fprintf('Cross validation round/fold number %d\n',Fold);
% % Select the vectors to use for training and cross validation.
% [X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% % Train svm using SMO
% fprintf('Training....\n');
% tic;
% svmstruct=svmtrain(X_train,y_train,'kernel_function','linear','method','SMO');
% t=toc;
% timeSMOlinear=timeSMOlinear+t;
% % Test svm on validation set
% pred=svmclassify(svmstruct,X_val);
% acc=(sum(pred==y_val)/size(y_val,1))*100;
% fprintf('Accuracy in fold %d=%f\n',Fold,acc);
% cvaccuracySMOlinear=cvaccuracySMOlinear+acc;
% fprintf('Time taken in fold %d=%f secs\n',Fold,t);
% end
% timeSMOlinear=timeSMOlinear/numFolds;
% cvaccuracySMOlinear=cvaccuracySMOlinear/numFolds;
% fprintf('Time taken per fold for the SMO linear kernel=%f\n',timeSMOlinear);
% fprintf('Classification Accuracy using SMO linear kernel is %f\n',cvaccuracySMOlinear);
% return;
%----UNCOMMENT FROM THE LINE ABOVE TO USE IMPLEMENTATION 2----


%----IMPLEMENTATION 3---
%---My SVM implementation(QP) with quadratic kernel---

%----UNCOMMENT FROM THE LINE BELOW TO USE IMPLEMENTATION 3----
% type=2;%quadratic kernel
% cvaccuracyQPquadratic=0;
% timeQPquadratic=0;
% for Fold=1:numFolds
% fprintf('Cross validation round/fold number %d\n',Fold);
% % Select the vectors to use for training and cross validation.
% [X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% % Train svm using QP
% fprintf('Be Patient!!.Training....\n');
% tic;
% [alpha,w_0]=train_svm(X_train,y_train,C,type);
% t=toc;
% timeQPquadratic=timeQPquadratic+t;
% % Test svm on validation set
% [pred,acc]=predict_svm(X_val,y_val,X_train,y_train,alpha,w_0,type);
% fprintf('Accuracy in fold %d=%f\n',Fold,acc);
% cvaccuracyQPquadratic=cvaccuracyQPquadratic+acc;
% fprintf('Fold %d time = %f\n',Fold,t);
% end
% timeQPquadratic=timeQPquadratic/numFolds;
% cvaccuracyQPquadratic=cvaccuracyQPquadratic/numFolds;
% fprintf('Time taken per fold for the QP quadratic kernel=%f\n',timeQPquadratic);
% fprintf('Classification Accuracy using QP quadratic kernel is %f\n',cvaccuracyQPquadratic);
% return;
%----UNCOMMENT FROM THE LINE ABOVE TO USE IMPLEMENTATION 3


%---IMPLEMENTATION 4---
%---Matlab SVM(SMO) quadratic

%---UNCOMMENT FROM THE LINE BELOW TO USE IMPLEMENTATION 4---
% cvaccuracySMOquadratic=0;
% timeSMOquadratic=0;
% for Fold=1:numFolds
% fprintf('Cross validation round/fold number %d\n',Fold);
% % Select the vectors to use for training and cross validation.
% [X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
% % Train svm using QP
% fprintf('Training....\n');
% tic;
% svmstruct=svmtrain(X_train,y_train,'kernel_function','quadratic','method','SMO');
% t=toc;
% timeSMOquadratic=timeSMOquadratic+t;
% pred=svmclassify(svmstruct,X_val);
% acc=(sum(pred==y_val)/size(y_val,1))*100;
% fprintf('Accuracy in fold %d=%f\n',Fold,acc);
% cvaccuracySMOquadratic=cvaccuracySMOquadratic+acc;
% fprintf('Fold %d time = %f\n',Fold,t);
% end
% timeSMOquadratic=timeSMOquadratic/numFolds;
% cvaccuracySMOquadratic=cvaccuracySMOquadratic/numFolds;
% fprintf('Time taken per fold for the SMO quadratic kernel=%f\n',timeSMOquadratic);
% fprintf('Classification Accuracy using SMO quadratic kernel is %f\n',cvaccuracySMOquadratic);
% return;
%---UNCOMMENT FROM THE LINE ABOVE TO USE IMPLEMENTATION 4---
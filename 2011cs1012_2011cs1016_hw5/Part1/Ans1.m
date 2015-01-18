load('dataset1');
 s=RandStream.create('mt19937ar','seed',54);
 RandStream.setDefaultStream(s);
T=500;%Number of Iterations
categories=[-1,1]';
vecsPerCat = getVecsPerCat(Y,categories);
numFolds=10;
foldSizes = computeFoldSizes(vecsPerCat, numFolds);
[X_sorted, y_sorted] = randSortAndGroup(X,Y,categories);
testaccfold=zeros(numFolds,1);
precisionfold=zeros(numFolds,2);
recallfold=zeros(numFolds,2);
MaxIter=500;
traintable=zeros(numFolds,MaxIter);
conffold=cell(numFolds,1);
for Fold=1:numFolds
    fprintf('Fold=%d\n',Fold);
    [X_train,y_train,X_val,y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
    n=size(X_train,1);%Number of training examples
    fprintf('Number of training examples %d\n',n);
    %---ADABOOST starts here
    %--Training---
    a_t=zeros(MaxIter,1);%alpha_t
    hyp=cell(MaxIter,1);%stores the hypothesis for every iteration
    D=(ones(n,1))/n;%initializing weights
    iter=MaxIter;
    for t=1:MaxIter
        indices=WeightedSampleWithReplacement(D,n);
        x=X_train(indices,:);
        y=y_train(indices,:);
        svmstruct=svmtrain(y,x,'-t 0 -q');
        hyp{t,1}=svmstruct;%store the learnt hypothesis
        [pred,~,~]=svmpredict(y,x,svmstruct,'-q');
        error=sum(D(pred~=y));
        %fprintf('error is %f\n',error);
        if error >= 0.5||error==0
            fprintf('Break Error is %f.\n',error);
            fprintf('Break Iteration=%d\n',t-1);
            iter=t-1;
            break;
        end
        a_t(t)=0.5*log((1-error)/error);
        D=D.*exp(-a_t(t)*(y.*pred));%Update weights
        D=D/sum(D);%Normalize updated weights
        %Calculate ensemble training accuracy till current iteration
        M=zeros(t,size(X_train,1));
        for k=1:t
            [pred,~,~]=svmpredict(y_train,X_train,hyp{k,1},'-q');
            M(k,:)=pred';
        end
        ensembpred=sign(M'*a_t(1:t,:));%ensembpred is a column vector of same size as y_val
        %size(ensembpred)
        %size(y_train)
        trainacc=((sum(ensembpred==y_train))/size(X_train,1))*100;
        %fprintf('Ensemble train_accuracy on iter %d=%f\n',t,trainacc);
        traintable(Fold,t)=trainacc;
    end
    %Ensemble predictions on the test set
    M=zeros(iter,size(X_val,1));
    for t=1:iter
        [pred,~,~]=svmpredict(y_val,X_val,hyp{t,1},'-q');
        M(t,:)=pred';
    end
    ensembpred=sign(M'*a_t(1:iter,:));%ensembpred is a column vector of same size as y_val
    testacc=(sum(ensembpred==y_val)/size(X_val,1))*100;
    fprintf('Ensemble accuracy on test set=%f\n',testacc);
    testaccfold(Fold)=testacc;%Test accuracy for each fold.
    %---ADABOOST ends here
    %---compute the confusion matrix for the test set here
    tp=sum(y_val==1&ensembpred==y_val);
    fn=sum(y_val==1)-tp;
    tn=sum(y_val==-1&ensembpred==y_val);
    fp=sum(y_val==-1)-tn;
    conf=[tp fn;fp tn];
    conffold{Fold}=conf;
    fprintf('truepositive=%f\nfalsenegative=%f\ntruenegative=%f\nfalsepositive=%f\n',tp,fn,tn,fp);
    fprintf('Precision for positive class=%f\n',tp/(tp+fp));
    fprintf('Precision for negative class=%f\n',tn/(tn+fn));
    precisionfold(Fold,:)=[tp/(tp+fp) tn/(tn+fn)];
    recallfold(Fold,:)=[tp/(tp+fn) tn/(tn+fp)];
end
avtrain=[];%stores the average training accuracy across folds
for i=1:MaxIter
    sum=0;
    count=0;
    for j=1:numFolds
        if traintable(j,i)~=0
            sum=sum+traintable(j,i);
            count=count+1;
        end
    end
    if sum~=0
        avtrain=[avtrain sum/count];
    end
end
s=0;
for i=1:length(testaccfold)
    s=s+testaccfold(i);
end
fprintf('-------Final Results------------\n');
avtest=s/numFolds;%average test accuracy
s=[0 0;0 0] ;
for i=1:length(conffold)
    s=s+conffold{i};
end
avconffold=s/numFolds;
s=[0 0];
for i=1:length(precisionfold)
    s=s+precisionfold(i,:);
end
precisionav=s/numFolds;
s=[0 0 ];
for i=1:length(recallfold)
    s=s+recallfold(i,:);
end
recallav=s/numFolds;
fprintf('Average training accuracy\n');
avtrain'
fprintf('Test accuracy per fold\n');
testaccfold
fprintf('Average test accuracy=\n');
avtest
fprintf('Average confusion matrix=\n');
avconffold
fprintf('Average precision of class1=%f\n',precisionav(1));
fprintf('Average precision of class2=%f\n',precisionav(2));
fprintf('Average recall of class1=%f\n',recallav(1));
fprintf('Average recall of class2=%f\n',recallav(2));
plot(1:length(avtrain),avtrain);
title('Training accuracy vs Iterations');
xlabel('Iterations');
ylabel('Training accuracy');
fprintf('Precision fold\n');
precisionfold
fprintf('Recall fold\n');
recallfold
fprintf('Confusion matrix fold\n');
for i=1:length(conffold)
    conffold{i}
end
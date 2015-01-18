load('mnist');
s=RandStream.create('mt19937ar','seed',5489);
RandStream.setDefaultStream(s);
Y=zeros(size(label,1),1);
%Convert label bitsets to nominal labels
n=size(label,1);
for i=1:n
Y(i)=find(label(i,:)==1);
end
categories=(1:10)';
%--------vecsPerCat(i) stores the number of examples with i as their category
vecsPerCat = getVecsPerCat(Y,categories);
numFolds=10;
%---foldSizes is a matrix #categories X #folds
%---foldSize(i,j) is number of examples belonging to 
%---ith category and jth fold
foldSizes = computeFoldSizes(vecsPerCat, numFolds);
%--------Sort them randomly
 X=data;
[X_sorted, y_sorted] = randSortAndGroup(X,Y,categories);
list=zeros(49,3);
num=1;
%Total time taken=7*7*300 secs=4hrs
for c=10.^(-3:3)
    for sigma=10.^(-3:3)
        cvaccuracy=0;
        tic;
        for Fold=1:numFolds%Takes 30 seconds per fold
            fprintf('Cross validation round/fold number %d,c=%f,g=%f\n',Fold,c,sigma);
            % Select the vectors to use for training and cross validation.
            [X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, y_sorted, categories, vecsPerCat, foldSizes, Fold);
            opts=['-c ' num2str(c) ' -g ' num2str(sigma) ' -q '];
            svmStruct=svmtrain(y_train,X_train,opts);
            [~,accuracy,~]=svmpredict(y_val,X_val,svmStruct);
            fprintf('Accuracy in fold %d=%f\n',Fold,accuracy(1));
            cvaccuracy=cvaccuracy+accuracy(1);
        end
    list(num,:)=[c sigma cvaccuracy/numFolds];
    t=toc;
    fprintf('Total time elapsed= %f secs',t);
    fprintf('Number of iteration=%d\n',num);
    num=num+1;
    end
end
for i=1:length(list)
    fprintf('c=%f,g=%f,cvaccuracy=%f',list(i,1),list(i,2),list(i,3));
end 
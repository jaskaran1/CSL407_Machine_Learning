function dataT=featuretransform(X,degree)
%Assuming that the data set has only 2 features
%and they are in columns
dataT=[];
for S=0:degree
    for x=0:S
    dataT=[dataT ((X(:,1).^(S-x)).*(X(:,2).^x))];
    end
end
end
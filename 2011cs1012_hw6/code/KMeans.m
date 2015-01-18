function [labels]=KMeans(X,K)
n=size(X,1);
d=size(X,2);
labels=randi(K,1,n);
centroids=zeros(K,d);
while true
    for i=1:K%%centroid computation
        clusterpoints=X(labels==i,:);%points belonging to the ith cluster
        centroids(i,:)=sum(clusterpoints)/size(clusterpoints,1);
    end
    newlabels=zeros(1,n);
    for i=1:n%centroid assignment step
        pt=X(i,:);
        min_dist=1000000000;%a large number
        for j=1:K
            dist=sqrt(sum((pt-centroids(j,:)).^2));
            if dist<min_dist
                min_dist=dist;
                newlabels(i)=j;
            end
        end
    end
    
    if labels==newlabels
          break;
    end
    
    labels=newlabels;
end

end
s=RandStream.create('mt19937ar','seed',54);
RandStream.setGlobalStream(s);
K=2;
n=6;
X=[1 1;1 3;0 4;6 2;5 1;4 0];
fprintf('Plot Data points\n');
plot(X(:,1),X(:,2),'ko','MarkerFaceColor','b','MarkerSize',7);%plotted the observations
pause;
hold off;
%cluster labels are 1 to K
labels=randi(K,1,n);%assign labels randomly
centroids=zeros(K,2);
fprintf('Plotting Randomized classified points\n');
labels
colormap=['b' 'g' 'r' 'c' 'm' 'y' 'k' 'w'];
for i=1:K
    classpoints=X(labels==i,:);
    plot(classpoints(:,1),classpoints(:,2),'ko','MarkerFaceColor',colormap(i),'MarkerSize',7);
    hold on;
end
pause;
while true
    for i=1:K%%centroid computation
        clusterpoints=X(labels==i,:);%points belonging to the ith cluster
        centroids(i,:)=sum(clusterpoints)/size(clusterpoints,1);
    end
    fprintf('Plotting Centroids\n');
    %plot the centroids
    for i=1:K
        plot(centroids(i,1),centroids(i,2),strcat(colormap(i),'+'));
        hold on;
    end
    hold off;
    pause;
    newlabels=zeros(1,n);
    for i=1:n%centroid assignment step
        pt=X(i,:);
        min_dist=100000000;%a large number
        for j=1:K
            dist=sqrt(sum((pt-centroids(j,:)).^2));
            if dist<min_dist
                min_dist=dist;
                newlabels(i)=j;
            end
        end
    end
    fprintf('Plotting Clustered points\n');
    for i=1:K
        classpoints=X(newlabels==i,:);
        plot(classpoints(:,1),classpoints(:,2),'ko','MarkerFaceColor',colormap(i),'MarkerSize',7);
        hold on;
    end
    
    if labels==newlabels
          break;
    end
    labels=newlabels;
end
fprintf('Resultant clusters\n');
labels
hold off;
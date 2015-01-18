img=698;%number of images
d=4096;%number of features
load('face_data');
data=images';
tic;
K=6;
%ADJ is the adjacency list.
%ADJ(i) contains the K+1 nearest neighbours.K+1 because the vertex itself
%is included in the neighbours list
[ADJ,~]=knnsearch(data,data,'K',K+1);%adjacency list of K nearest neigbours
Dg=zeros(img,img);
%type='euclidean';%Uncomment for euclidean
type='cosine';%Uncomment for cosine distance
for i=1:img
    list=ADJ(i,:);
    for j=1:length(list)
        Dg(i,list(j))=pdist([ data(i,:) ; data(list(j),:) ],type);%distance between the ith data point and its nearest neighbour
        Dg(list(j),i)=Dg(i,list(j));
    end
end
t=toc;
fprintf('NBD graph created in time=%f\n',t);
%Calculate the shortest paths.O(n^3)
tic;
Dist=Dg;
Dg=graphallshortestpaths(sparse(Dg));
t=toc;
fprintf('Johnson=%f secs\n',t);
%Construct the Gram Matrix
Gram=zeros(img,img);
Gram=-0.5*(Dg-mean(Dg,2)*ones(1,img)-ones(img,1)*mean(Dg,1)+mean(Dg(:)));
[U,lambda,~]=svd(Gram);
K=2;%reduced dimensions
U_red=U(:,1:K);%U_red is NxK.Each column is eig vector
data_red=zeros(img,K);%data in reduced dimensions
for i=1:img
    r=zeros(1,K);
    for j=1:K
        r(j)=sqrt(lambda(i,i))*U_red(i,j);
    end
    data_red(i,:)=r;
end
figure(1);
plot(data_red(:,1),data_red(:,2),'.');
[B,I]=sortrows(data_red);
dispimages(I,data,data_red);
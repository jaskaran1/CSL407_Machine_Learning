s=RandStream.create('mt19937ar','seed',59);
RandStream.setGlobalStream(s);
C=3;%3 classes
n=20;%data points per class
d=50;
origlabels=[];
for i=1:C
    origlabels=[origlabels;i*ones(n,1)];
end
origlabels=origlabels';
%X=rand(n,d);random doesn't work
cent=zeros(C,d);%centroid of classes
for i=1:C
    cent(i,:)=5*i*ones(1,d);
end
X=[];
sd=5;
for i=1:C
    class=randn(n,d)*sd+ones(n,1)*cent(i,:);%normalized data for class
    X=[X;class];
end
u=mean(X,1);
Xdemeaned=X-ones(C*n,1)*u;
Sigma=Xdemeaned'*Xdemeaned/(C*n);%dxd matrix
[U,S,~]=svd(Sigma);
K=2;%number of reduced dimensions
U_red=U(:,1:K);
U_red=U_red';%Projection matrix of size Kxd
X_red=U_red*X';%X in the new projections space of K dimensions
X_red=X_red';%X_red is a matrix of nxK size
plot(X_red(1:n,1),X_red(1:n,2),'ko','MarkerFaceColor','r');
hold on;
plot(X_red(n+1:2*n,1),X_red(n+1:2*n,2),'ko','MarkerFaceColor','g')
hold on;
plot(X_red(2*n+1:3*n,1),X_red(2*n+1:3*n,2),'ko','MarkerFaceColor','b')
hold off;

labels=KMeans(X,3);%(c)
fprintf('K=3 on original data\n');
fprintf('Original labels\n');
origlabels
fprintf('Clustered Labels\n');
labels
pause;
labels=KMeans(X,2);%(d)
fprintf('K=2 on original data\n');
fprintf('Original labels\n');
origlabels
fprintf('Clustered Labels\n');
labels
pause;
labels=KMeans(X,4);%(e)
fprintf('K=4 on original data\n');
fprintf('Original labels\n');
origlabels
fprintf('Clustered Labels\n');
labels
pause;
labels=KMeans(X_red,3);%(f)
fprintf('K=3 on projected data\n');
fprintf('Original labels\n');
origlabels
fprintf('Clustered Labels\n');
labels
pause;
X_Var=X./(ones(C*n,1)*var(X));%(g)
labels=KMeans(X_Var,3);
fprintf('K=3 on standardized data\n');
fprintf('Original labels\n');
origlabels
fprintf('Clustered Labels\n');
labels
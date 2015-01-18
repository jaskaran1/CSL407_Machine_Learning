function lindiscriminant(X,y,deg)
%2 classes y=1 and y=0
%Taking 1 also as a feature
[n,d]=size(X);
train1=X(y==1,:);
u1=mean(train1);
u1_mat=repmat(u1,size(train1,1),1);
train0=X(y==0,:);
pi0=size(train0,1)/size(X,1);
pi1=size(train1,1)/size(X,1);
u0=mean(train0);
u0_mat=repmat(u0,size(train0,1),1);
train1=(train1-u1_mat);
train0=(train0-u0_mat);
cov=zeros(d,d);
for i=1:size(train0,1)
    cov=cov+train0(i,:)'*train0(i,:);
end
for i=1:size(train1,1)
    cov=cov+train1(i,:)'*train1(i,:);
end
cov=cov./(n-2);
x1 = linspace(0, 10, 50);
x2 = linspace(0, 10, 50);
z = zeros(length(x1),length(x2));
% Evaluate z = w*x over the grid
for i = 1:length(x1)
        for j = 1:length(x2)
            a=mapFeature(x1(i), x2(j),deg);
            z(i,j) =-(0.5)*(u1/cov*u1'-u0/cov*u0')+log(pi1/pi0)+a(:,2:end)/cov*(u1-u0)';
        end
end
z = z'; % important to transpose z before calling contour
    % Plot z = 0
    % Notice you need to specify the range [0, 0]
contour(x1,x2, z, [0,0],'c', 'LineWidth', 2);

end
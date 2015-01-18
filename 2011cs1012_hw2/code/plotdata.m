function plotdata(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
figure;
hold on;
I1=find(y==1);%used to find the indices of the array y where y==1 
I0=find(y==0);
plot(X(I1,1),X(I1,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(I0,1),X(I0,2),'ko','MarkerFaceColor','r','MarkerSize',7);
hold off;
end

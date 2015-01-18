t1=normrnd(2,.2,10,1);%Column of 10 data points each
t2=normrnd(3,.2,10,1);
t3=normrnd(4,.2,10,1);
tr_row=[t1'; t2'; t3'];
train=[t1;t2;t3];%column of data points
x=1:0.05:5;
test=(1:0.1:5)';%test column data
z=zeros(1,10);
scatter(t1,z,'r');
hold on;
scatter(t2,z,'g');
scatter(t3,z,'b');
u1=mean(t1);
sig1=std(t1);
u2=mean(t2);
sig2=std(t2);
u3=mean(t3);
sig3=std(t3);
Y1=normpdf(x,u1,sig1);
Y2=normpdf(x,u2,sig2);
Y3=normpdf(x,u3,sig3);
a1=Y1.*(length(t1));
a2=Y2.*(length(t2));
a3=Y3.*(length(t3));
res=plot(x,Y1,'r',x,Y2,'g',x,Y3,'b',x,a1,'r:',x,a2,'g:',x,a3,'b:');
legend('ClassY1','ClassY2','ClassY3','P(X=x|Y=Y1)','P(X=x|Y=Y2)','P(X=x|Y=Y3)','P(Y=Y1|X=x)','P(Y=Y2|X=x)','P(Y=Y3|X=x)','Location','SouthEast');
hold off;
pi1=length(t1)/length(train);
pi2=length(t2)/length(train);
pi3=length(t3)/length(train);
cov=0;
u=[u1 u2 u3];
for i=1:3%Did it in this way since data is one-dimensional
    cov=cov+sum((tr_row(i,:)-u(i)).^2);
end
cov=cov/(length(train)-3);
figure;
%color coded plot of test data
disc1=test.*(1/cov)*u1-.5*u1*(1/cov)*u1+log(pi1);%LDAtest 1
disc2=test.*(1/cov)*u2-.5*u2*(1/cov)*u2+log(pi2);%LDAtest 2
disc3=test.*(1/cov)*u3-.5*u3*(1/cov)*u3+log(pi3);%LDAtest 3
plot(test,disc1,'r',test,disc2,'g',test,disc3,'b');%plot of LDAs
disc=[disc1 disc2 disc3];
hold on;
test1=[];
test2=[];
test3=[];
for i=1:length(test)
[~,j]=max(disc(i,:));
 if j==1
 test1=[test1 test(i)];
 end
 if j==2
 test2=[test2 test(i)];
 end
 if j==3
 test3=[test3 test(i)];
 end
end
scatter(test1,zeros(1,size(test1,2)),'r');
scatter(test2,zeros(1,size(test2,2)),'g');
scatter(test3,zeros(1,size(test3,2)),'b');
scatter(t1,zeros(1,length(t1)),'r','fill');
scatter(t2,zeros(1,length(t2)),'g','fill');
scatter(t3,zeros(1,length(t3)),'b','fill');
legend('LDA1','LDA2','LDA3','Y1(test)','Y2(test)','Y3(test)','Y1(train)','Y2(train)','Y3(train)','Location','SouthEast');
title('Classification of test data using LDA');
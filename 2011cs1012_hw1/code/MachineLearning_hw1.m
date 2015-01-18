load('abalone.mat');
[n,d]=size(abalone);
X=abalone;
first=X(:,1);
female=first==0;
infant=first==1;
male=first==2;
X=X(:,2:end);%Removed first col
X=[female infant male X];
%Remove the visceral weight(9th col).Least significant
X=X(:,[1:3,6:end]);
reps=100;
lambda_low=0;
lambda_step=.1;
lambda_high=1;
frac_low=0.2;
frac_step=0.2;
frac_high=0.8;
Y_train_min_err=[];
Y_test_min_err=[]; 
lambda_train_min=[];
lambda_test_min=[];
for frac=frac_low:frac_step:frac_high%Fraction of split for train/test
    %fprintf('Fraction_split=%f\n',frac);
    Y_train_err=[];
    Y_test_err=[];
    L=lambda_low:lambda_step:lambda_high;
    L2=frac_low:frac_step:frac_high;
    for lambda=lambda_low:lambda_step:lambda_high%ridge regression parameter
        %fprintf('\nlambda=%f\n',lambda);
        average_train_error=0;
        average_test_error=0;   
        for i=1:reps%For given fraction and lambda iterate reps times
            index=randperm(n);
            X_rand=X(index,:);%Randomized
            num_rows=floor(frac*n);
            X_train=X_rand(1:num_rows,:);%Training
            T_train=X_train(:,end);%Training set age
            X_train=X_train(:,1:end-1);%Truncate Training set of age
            u=mean(X_train);
            u_mat=repmat(u,size(X_train,1),1);
            sigma=std(X_train);
            sigma_mat=repmat(sigma,size(X_train,1),1);
            X_train=(X_train-u_mat)./sigma_mat;%Standardize training set
            
            X_test=X(num_rows+1:end,:);%Test
            T_test=X_test(:,end);%Test set age
            X_test=X_test(:,1:end-1);%Truncate Test set of age
            u_mat=repmat(u,size(X_test,1),1);
            sigma_mat=repmat(sigma,size(X_test,1),1);
            X_test=(X_test-u_mat)./sigma_mat;%Standardize test set
            
            W=mylinridgereg(X_train,T_train,lambda);%Train
            W
            [~,I]=min(abs(W));
            fprintf('Least sig feature w%d\n',I);
            train_pred=mylinridgeregeval(X_train,W);
            test_pred=mylinridgeregeval(X_test,W);
            rms_train_error=sqrt(meansquarederr(T_train,train_pred));
            rms_test_error=sqrt(meansquarederr(T_test,test_pred));
            %fprintf('lambda=%f,rms_train_error=%f,rms_test_error=%f\n',lambda,rms_train_error,rms_test_error);
            average_train_error=average_train_error + rms_train_error.^2;
            average_test_error=average_test_error + rms_test_error.^2;
        end
        average_train_error=average_train_error/reps;
        average_test_error=average_test_error/reps;
        %fprintf('average_train_error=%f\n',average_train_error);
        %fprintf('average_test_error=%f\n\n\n',average_test_error);
        Y_train_err=[Y_train_err average_train_error];
        Y_test_err=[Y_test_err average_test_error];
    end
    [MIN,I]=min(Y_train_err);
    lambda_train_min=[lambda_train_min lambda_low+(I-1)*lambda_step];
    Y_train_min_err=[Y_train_min_err MIN];
    [MIN,I]=min(Y_test_err);
    lambda_test_min=[lambda_test_min lambda_low+(I-1)*lambda_step];
    Y_test_min_err=[Y_test_min_err MIN];
    %Graphs
    subplot(1,(frac_high-frac_low)/frac_step+1,frac/frac_step);
    plot(L,Y_train_err,'-x',L,Y_test_err,'-o');
    title(sprintf('Frac=%f',frac));
    xlabel('Lambda');
    ylabel('Average MSE');
    ylim([3,5]);
    legend('TrainingMSE','TestMSE','Location','SouthEast');
    
end        
figure;
subplot(1,2,1);
%plot(L2,Y_train_min_err,'-x',L2,Y_test_min_err,'-o');
plot(L2,Y_test_min_err,'-x');
xlabel('Training/Test split');
ylabel('Min average TestMSE');
legend('MinTestMSE','Location','SouthEast');
subplot(1,2,2);
plot(L2,lambda_test_min,'-x');
xlabel('Training/Test split');
ylabel('Lambda corresponding to minTestMSE');
legend('LambdaforTest','Location','SouthEast');
[MIN,I]=min(Y_test_min_err);
frac=frac_low+(I-1)*frac_step;%Set frac
lambda_min=lambda_test_min(I);%Set lambda
index=randperm(n);
X_rand=X(index,:);%Randomized
num_rows=floor(frac*n);
X_train=X_rand(1:num_rows,:);%Training
T_train=X_train(:,end);%Training set age
X_train=X_train(:,1:end-1);%Truncate Training set of age
u=mean(X_train);
u_mat=repmat(u,size(X_train,1),1);
sigma=std(X_train);
sigma_mat=repmat(sigma,size(X_train,1),1);
X_train=(X_train-u_mat)./sigma_mat;%Standardize training set
           
X_test=X(num_rows+1:end,:);%Test
T_test=X_test(:,end);%Test set age
X_test=X_test(:,1:end-1);%Truncate Test set of age
u_mat=repmat(u,size(X_test,1),1);
sigma_mat=repmat(sigma,size(X_test,1),1);
X_test=(X_test-u_mat)./sigma_mat;%Standardize test set
            
W=mylinridgereg(X_train,T_train,lambda);%Train
train_pred=mylinridgeregeval(X_train,W);
test_pred=mylinridgeregeval(X_test,W);
refx=linspace(-5,25,20);
refy=refx;
figure;
subplot(1,2,1);
plot(train_pred,T_train,'x',refx,refy,'-r');
xlabel('Training Prediction');
ylabel('Training Actual');
axis([-5 25 -5 25]);
subplot(1,2,2);
plot(test_pred,T_test,'x',refx,refy,'-r');
title(sprintf('Split Fraction=%f,Lambda=%f',frac,lambda_min));
xlabel('Testing Prediction');
ylabel('Testing Actual');
axis([-5 25 -5 25]);
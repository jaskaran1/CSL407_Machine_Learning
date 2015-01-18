function hw3part1e
close all;
clear all;
clc;
% we will experiment with a simple 2d dataset to visualize the decision
% boundaries learned by a MLP. Our goal is to study the changes to the
% decision boundary and the training error with respect to the following
% parameters
% - increasing overlap between the data points of the different classes
% - increasing the number of training iterations
% - increase the number of hidden layer neurons
% - see the effect of learning rate on the convergence of the network


% centroid for the three classes
c1=[1 1];
c2=[3 1];
c3=[2 3];

% standard deviation for the three classes
% "increase this quantity to increase the overlap between the classes"
% change this quantity to 0.75 when solving 1(f).
sd=0.2;

% number of data points per class
N=100;

rand('seed', 1);

% generate data points for the three classes
%x1,x2 and x3 are each Nx2 matrices
x1=randn(N,2)*sd+ones(N,1)*c1;
x2=randn(N,2)*sd+ones(N,1)*c2;
x3=randn(N,2)*sd+ones(N,1)*c3;

% generate the labels for the three classes in the binary notation
y1= repmat([1 0 0],N,1);
y2= repmat([0 1 0],N,1);
y3= repmat([0 0 1],N,1);

% creating the test data points
a1min = min([x1(:,1);x2(:,1);x3(:,1)]);
a1max = max([x1(:,1);x2(:,1);x3(:,1)]);

a2min = min([x1(:,2);x2(:,2);x3(:,2)]);
a2max = max([x1(:,2);x2(:,2);x3(:,2)]);

[a1 a2] = meshgrid(a1min:0.1:a1max, a2min:0.1:a2max);

testX=[a1(:) a2(:)];

% Experimenting with MLP

% number of epochs for training
nEpochs = 1000;


% number of hidden layer units
H = 16;

% train the MLP using the generated sample dataset
figure;
for eta=[.001 .01 .1]
[w, v, trainerror] = mlptrain([x1;x2;x3],[y1;y2;y3], H, eta, nEpochs);

% plot the train error againt the number of epochs
 plot(1:nEpochs, trainerror, 'LineWidth', 2);
 hold all;
end
legend('eta=0.001','eta=0.01','eta=0.1','Location','North');
function [w v trainerror] = mlptrain(X, Y, H, eta, nEpochs)
% X - training data of size NxD
% Y - training labels of size NxK
% H - the number of hidden units
% eta - the learning rate
% nEpochs - the number of training epochs
% define and initialize the neural network parameters

% number of training data points
N = size(X,1);
% number of inputs
D = size(X,2); % excluding the bias term
% number of outputs
K = size(Y,2);

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% w is a Hx(D+1) matrix
w = -0.3+(0.6)*rand(H,(D+1));

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% v is a Kx(H+1) matrix
v = -0.3+(0.6)*rand(K,(H+1));

% randomize the order in which the input data points are presented to the
% MLP
iporder = randperm(N);
trainerror=zeros(1,nEpochs);
% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    for n = 1:N
        % the current training point is X(iporder(n), :)
        % forward pass
        % --------------
        % input to hidden layer
        % calculate the output of the hidden layer units - z
        % ---------
        %'TO DO'%
        tp=X(iporder(n),:);%adding the bias term to training point
        t=Y(iporder(n),:);%training label of tp
        z=zeros(H,1);
        z=1./(1.+exp(-w*[1 tp]'));%sigmoid
        % ---------
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        %'TO DO'%
        ydash=zeros(K,1);
        ydash=exp(v*[1;z])/sum(exp(v*[1;z]));%softmax function
        % ---------
        
        % backward pass
        % ---------------
        % update the weights for the connections between hidden and
        % outlayer units
        % ---------
        %'TO DO'%
        %cross entropy is the error function
        deltaV=eta*(t'-ydash)*[1 z'];
        v=v+deltaV;
        % ---------
        % update the weights for the connections between the input and
        % hidden later units
        % ---------
        %'TO DO'%
        deltaW=(eta*v(:,2:end)'*(t'-ydash)*ones(1,D+1)).*((z.*(1-z))*[1 tp]);
        w=w+deltaW;
        % ---------
    end
    ydash = mlptest(X, w, v);%ydash has the dimension NxK
    % compute the training error
    % ---------
    %'TO DO'% uncomment the next line after adding the necessary code
    trainerror(epoch) =-sum(sum(Y.*log(ydash)));
    % ---------
    disp(sprintf('training error after epoch %d: %f\n',epoch,...
        trainerror(epoch)));
end
return;

function ydash = mlptest(X, w, v)
% forward pass of the network

% number of inputs
N = size(X,1);

% number of outputs
K = size(v,1);
H = size(w,1);
ydash=zeros(N,K);
for i=1:N
% forward pass to estimate the outputs
% --------------------------------------
% input to hidden for all the data points
% calculate the output of the hidden layer units
% ---------
%'TO DO'%
tp=X(i,:);
z=zeros(H,1);
z=1./(1.+exp(-w*[1 tp]'));%sigmoid
% ---------% hidden to output for all the data points
% calculate the output of the output layer units
% ---------
%'TO DO'%
res=zeros(K,1);
res=exp(v*[1;z])/sum(exp(v*[1;z]));%softmax function
ydash(i,:)=res';
end
% ---------
return;
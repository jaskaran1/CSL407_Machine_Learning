function hw3part4vectorized

load('mnist.mat');

tridx = [];
validx = [];
tsidx = [];

% fraction of the data to be used for validation and test.
valfrac = 0.2;
tsfrac = 0.2;

for k = 1:size(label,2)
    % find the indices of the data points of a particular class
    r = find(label(:,k) == 1);
    % number of data points belonging to the kth class
    nclass = length(r);
    % randomize the indices for the k^{th} class data points
    ridx = randperm(nclass);
    % use the first nclass*tsfrac indices as the test data points
    temptsidx = r(ridx(1:nclass*tsfrac));
    % use the next nclass*valfrac indices as the validation set
    tempvalidx = r(ridx(nclass*tsfrac+1:nclass*tsfrac+1+nclass*valfrac));
    % use the remaining indices as training points
    temptridx = setdiff(r, [temptsidx; tempvalidx]);    
    % append the indices to the cumulative variable
    tridx = [tridx temptridx];
    tsidx = [tsidx temptsidx];
    validx = [validx tempvalidx];
end

% separate the train, validation and test datasets
trX = data(tridx,:);
trY = label(tridx,:);

valX = data(validx,:);
valY = label(validx,:);

tsX = data(tsidx,:);
tsY = label(tsidx,:);


% size of the training data
[N, D] = size(trX);
% number of epochs
nEpochs = 100;
% learning rate
eta = 0.01;
% number of hidden layer units
H = 500;
% number of output layer units
K = 10;

% randomize the weights from input to hidden layer units
% 'TO DO'
 w = -0.3+(0.6)*rand(H,(D+1));
% randomize the weights from hidden to output layer units
% 'TO DO'
 v = -0.3+(0.6)*rand(K,(H+1));

% let us create the indices for the batches as it cleans up script later
% size of the training batches
batchsize = 25;
% number of batches
nBatches = floor(N/batchsize);
% create the indices of the data points used for each batch
% i^th row in batchindices will give th eindices for the data points for
% the i^th batch
batchindices = reshape([1:batchsize*nBatches]',batchsize, nBatches);
batchindices = batchindices';
% if there are any data points left out, add them at the end padding with
% some other indices from the previous batch
if N - batchsize*nBatches >0
    batchindices(end+1,:)=batchindices(end,:);
    batchindices(end,1:(N - batchsize*nBatches)) = [batchsize*nBatches+1: N];
end

% randomize the order of the training data
ridx = randperm(N);
trX = trX(ridx,:);
trY= trY(ridx,:);

train_err=[];
val_err=[];
trainerror_epoch=zeros(1,nEpochs);
valerror_epoch=zeros(1,nEpochs);
fprintf('Total Number of training examples=%d',N);
for epoch = 1:nEpochs
    fprintf('Epoch_%d\n',epoch);
    fprintf('Number of batches=%d',nBatches);
    for batch = 1:nBatches
        % Call the forward pass function to obtain the outputs
        % 'TO DO'
%         fprintf('Size of training batch');
%         size(trX(batchindices(batch,:),:))
        fprintf('Epoch_%d,Batch_%d\n',epoch,batch);
        batch_size=length(batchindices(batch,:));
        fprintf('Batch_%d size=%d\n',batch,batch_size);
        [z ydash] = forwardpass(trX(batchindices(batch,:),:), w, v);%z is layer of hidden units(Hx1) 
%         fprintf('Size of w');
%         size(w);
%         fprintf('Size of v');
%         size(v);
        %and ydash has size(trX(batchindices(batch,:),:),1) x K
        %z is a matrix of size NxH  where i^th row contains the value of
        %the hidden layer for the i^th training example
        %To vectorize forwardpass
        % Call the gradient function to obtain the required gradient
        % updates
        % 'TO DO'
        [deltaw deltav] = computegradient(trX(batchindices(batch,:),:), trY(batchindices(batch,:),:), w, v, z, ydash);
        %To vectorize compute gradient
        % update the weights of the two sets of weights
        % 'TO DO'
        w = w - (eta*deltaw)./batch_size;
        v = v - (eta*deltav)./batch_size;
        % at the end of epoch compute the classification error on training
        % and validation dataset
        % 'TO DO'
    end
    
    %the weights have been trained
    [z ydash] = forwardpass(trX, w, v);%running on the training set
  %  trainerror_epoch(epoch)=-sum(sum(trY.*log(ydash)));
     error = classerror(trY, ydash);
     fprintf('Misclasssified examples out of %d = %d',size(trX,1),error);
     fprintf('Training accuracy after epoch_%d is %f\n',epoch,(1-error/size(trX,1))*100);
     train_err=[train_err , error];
    [z ydash] = forwardpass(valX, w, v);%running on the test set
%    valerror_epoch(epoch)=-sum(sum(valY.*log(ydash)));
     error = classerror(valY, ydash);
     val_err=[val_err , error];
     fprintf('Misclasssified examples out of %d = %d',size(valX,1),error);
     fprintf('Validation accuracy after epoch_%d is %f\n',epoch,(1-error/size(valX,1))*100);
end
%Plot of training and validation error wrt epochs
figure;
plot(1:nEpochs,train_err,'r-*',1:nEpochs,val_err,'g-*');
title('Number of Misclassified instances vs Number of Epochs');
legend('Training Misclassified','Test Misclassified','Location','NorthEast');
figure;
% compute the classification error on the test set
% 'TO DO'
 [z ydash] = forwardpass(tsX, w, v);%running on the test set
 error=classerror(tsY, ydash);
fprintf('\n\n----Misclassified examples on Test set is %d out of %d---\n\n',error,size(tsX,1));
fprintf('\n\n----Test accuracy is %f---\n\n',(1-error/size(tsX,1))*100);

% plot atmost 2 misclassified examples for each digit using the displayData
% function
misclass=[];
[~,ind]=max(ydash,[],2);%ind is Nx1->each row contains index of highest element
ydash=zeros(size(ydash));
for i=1:length(ind)
ydash(i,ind(i))=1;
end
%ydash
%size(ydash)
for k=1:K
    no=0;
    for i=1:N
        if no<=1 && tsY(i,k)==1 && ydash(i,k)==0%training example with label k but misclassified
            fprintf('label of example=%d\n',k);
            fprintf('predicted label of example=%d\n',find(ydash(i,:)==1));
            misclass(end+1,:)=tsX(i,:);
            no=no+1;
        end
    end
end
displayData(misclass);

function [z ydash] = forwardpass(X, w, v)
% this function performs the forward pass on a set of data points in the
% variable X and returns the output of the hidden layer units- z and the
% output layer units ydash
% 'TO DO'
%number of inputs
N=size(X,1);
% number of outputs
K = size(v,1);
H = size(w,1);
ydash=zeros(N,K);
z=zeros(N,H);
%To vectorize loop
z=tanh([ones(N,1) X]*w');%NxH
res=exp([ones(N,1) z]*v');
ydash=res./(sum(res,2)*ones(1,size(res,2)));
% [~,ind]=max(ydash,[],2);%ind is Nx1->each row contains index of highest element
% yres=zeros(size(ydash));
% for i=1:length(ind)
% yres(i,ind(i))=1;
% end

return;

function [deltaw deltav] = computegradient(X, Y, w, v, z, ydash)
% this function computes the gradient of the error function with resepct to
% the weights
% 'TO DO'
N=size(X,1);
D=size(X,2);
%size(w)
%size(v)
deltaw=zeros(size(w));%Hx(D+1)
deltav=zeros(size(v));%Kx(H+1)
%z is NxH
deltav=-((Y-ydash)'*[ones(size(z,1),1) z]);%ydash is NxK,z is NxH
deltaw=-(((v(:,2:end)'*(Y-ydash)').*(1-(z').^2))*[ones(N,1) X]);%vectorise this
return;

function error = classerror(y, ydash)
% this function computes the classification error given the actual output y
% and the predicted output ydash
[~,ind]=max(ydash,[],2);%ind is Nx1->each row contains index of highest element
ydash=zeros(size(ydash));
for i=1:length(ind)
ydash(i,ind(i))=1;
end
error = sum(sum(abs(y-ydash), 2)>0);
return;

function [h, display_array] = displayData(X)
% DO NOT CHANGE ANYTHING HERE
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.
%   example to plot the data provided by Andrew Ng.

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;
return;
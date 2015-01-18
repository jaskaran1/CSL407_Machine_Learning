function Ans4b
load('mnist');
s=RandStream.create('mt19937ar','seed',5489);
RandStream.setDefaultStream(s);
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

[N, D] = size(trX);

% randomize the order of the training data
ridx = randperm(N);
trX = trX(ridx,:);
trY= trY(ridx,:);

%---------MLP code starts here(from hw3)------------

% size of the training data
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

%Training loop
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
end
%the weights have been trained
[z ydash] = forwardpass(valX, w, v);%running on the validation set
%here ydash contains real numbers->round to 0 or 1
[error,ydash] = classerror(valY, ydash);%the ydash is a bit string
fprintf('Misclasssified examples out of %d = %d',size(valX,1),error);
fprintf('Validation accuracy is %f\n',(1-error/size(valX,1))*100);
%In case of Fmicro,FP=FN
%So,Fmicro=tp/N
tp=sum(sum(ydash==1&valY==1));%number of true positives
Fmlp=tp/size(valY,1);%micro-f1

%---------MLP code ends here-------------------------



%---------SVM code starts here-----------------------
Y_train=zeros(N,1);
%Convert label bitsets of training set to nominal labels
for i=1:N
Y_train(i)=find(trY(i,:)==1);
end
Y_val=zeros(size(valY,1),1);
for i=1:size(valY,1)
Y_val(i)=find(valY(i,:)==1);
end
opts=['-c ' num2str(10) ' -g ' num2str(0.1) ' -q '];
svmStruct=svmtrain(Y_train,trX,opts);
[ydash,accuracy,~]=svmpredict(Y_val,valX,svmStruct);
fprintf('Validation Accuracy=%f\n',accuracy(1));
tp=sum(ydash==Y_val);
Fsvm=tp/length(Y_val);
%-------------Results---------------
fprintf('F value of mlp=%f\n',Fmlp);
fprintf('F value of svm=%f\n',Fsvm);
%-------Functions used in MLP code-----------------
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

function [error,ydash] = classerror(y, ydash)
% this function computes the classification error given the actual output y
% and the predicted output ydash
[~,ind]=max(ydash,[],2);%ind is Nx1->each row contains index of highest element
ydash=zeros(size(ydash));
for i=1:length(ind)
ydash(i,ind(i))=1;
end
error = sum(sum(abs(y-ydash), 2)>0);
return;
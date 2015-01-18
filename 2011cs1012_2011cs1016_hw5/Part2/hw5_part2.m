%HW5  Machine Learning(ML)


%Clearing the variables in workspace environment
close all;
clear all;

%Clearing the workspace screen
clc;

%The number of weak classifier(tree)
NumWeakClassifier = 10;

%Loading the dataset
load 'recvstalkmini.mat';
rng('default');

%Converting class labels --required, as given in the paper {-1, 1} ----->  {0, 1}, according to the TrAdaBoost paper
trlabel(find(trlabel == -1)) = 0;
tslabel(find(tslabel == -1)) = 0;

%The magic number, required due to RAM constraints, increase this value if
%your system has enough RAM (>= 4GB).
magic_number = 50;

%The proportion taken from source dataset
ratio_source = 0.05;

%The proportion taken from target dataset
ratio_target = 0.05;

%Accuracy vector
accuracy = zeros(10,1);

%Standard deviation vector
sd = zeros(10,1);

%number of rows in source dataset
snewsize = size(tsdata, 1);

%number of rows in target dataset
tsize = size(trdata, 1);

%The loop counter
jj = 1;

%The random count variable, required by rng()
random_count = 1;

%The random folds
while(jj <= 10)
    %Using rng()
    rng(19 + random_count*2);

    %Using randperm()
    snew = randperm(snewsize);
    snewt = snew(1 : ceil(ratio_source * snewsize));

    %Using randperm()
    t = randperm(tsize);
    tnew = t(1 : ceil(ratio_target * tsize));

    %Getting trainX dataset
    trainX = [full(trdata(tnew, :)); full(tsdata(snewt, :))];

    %Getting trainY labels
    trainY = [trlabel(tnew, :); tslabel(snewt)];

    %Getting totalnum and roundnum, required due to RAM constraints
    totalnum = length(t(ceil(ratio_target * tsize) + 1 : end));
    roundnum = ceil(totalnum / magic_number) - 1;

    %Getting the class labels for validation dataset
    validationY = trlabel(t(ceil(ratio_target * tsize) + 1 : end));

    %Calculating beta, which is the constant factor by which the weights of
    %diff-distribution datasets are to be reduced
    beta  = (1+((2*log(length(snewt)/NumWeakClassifier))^(0.5))) ^ (-1);

    %Initialising beta_t, required to increase the weights of
    %same_distribution dataset
    beta_t = zeros(1,NumWeakClassifier);

    %The final outcome vector, the classification
    finalOutcome = zeros(totalnum, 1);

    %Outcome matrix for same-distribution dataset
    predict_tr = ones(length(tnew), NumWeakClassifier);

    %Outcome matrix for diff-distribution dataset
    predict_ts = ones(length(snewt), NumWeakClassifier);

    %Outcome matrix for test(validation, here) dataset
    predict_validation = ones(totalnum, NumWeakClassifier);

    %Intial weights vector with equal probabilities assigned
    w = ones(size(trainX, 1), 1) * ((size(trainX, 1))^(-1));

    %The weak classifier loop
    for i=1:NumWeakClassifier

        %Getting the optimum tree everytime using weights, which is passed
        %to Matlab's built-in function for decision-stumps, classregtree
        tree_output = classregtree(trainX, trainY, 'minparent', size(trainX, 1), 'MergeLeaves', 'off', 'weights', w);

        %Calculting the outcome of same-distribution dataset
        predict_tr(:, i) = eval(tree_output, trainX(1:length(tnew),:));

        %Calculating the error on same-distribution dataset
        error = sum((w(1 : length(tnew)) .* (((predict_tr(:, i) - trainY(1:length(tnew))) .^ 2) .^ 0.5)) / sum(w(1 : length(tnew))));

        %Calculating beta_t, required for same-distribution samples
        beta_t(i) = error * ((1 - error) ^ (-1));

        %Necessary computational trick, as log(beta_t) is required in the last
        %step, and log(0) is not defined
        if(beta_t(i) == 0)
            beta_t(i) = 0.001001;
        end

        %Updating the weights for same-distribution dataset
        for j = 1 : length(tnew)
            w(j) = w(j) * beta_t(i) ^ (-1 * (((predict_tr(j,i) - trainY(j)) .^ 2) .^ 0.5));
        end

        %Calculting the outcome of diff-distribution dataset
        predict_ts(:, i) = eval(tree_output, trainX(length(tnew)+1:end, :));

        %Updating the weights for diff-distribution dataset
        for j = 1 : length(snewt)
            w(length(tnew) + j) = w(length(tnew) + j) * beta ^ (((predict_ts(j, i) - trainY(length(tnew) + j)) ^ 2) ^ 0.5);
        end

        %This is required for temporary storage of the outcome of
        %validation samples, required due to computational constraints(RAM, or physical memory constraints)
        predict_validation_temp = [];

        %Calculating the outcome of validationX
        for(icount = 1 : roundnum)
            tvalidation = t(ceil(ratio_target * tsize) + magic_number * (icount - 1) + 1 : ceil(ratio_target * tsize) + magic_number * icount);
            validationX = full(trdata(tvalidation, :));
            predict_validation_temp = [predict_validation_temp; eval(tree_output, validationX)];
            clear validationX;
        end

        %Calculating the outcome of remaining validationX
        if((totalnum - roundnum*magic_number) ~= 0)
            tvalidation = t(ceil(ratio_target * tsize) + magic_number * roundnum + 1: end);
            validationX = full(trdata(tvalidation, :));
            predict_validation_temp = [predict_validation_temp; eval(tree_output, validationX)];
            clear validationX;
        end

        %Getting the predict_validation matrix
        predict_validation(:, i) = predict_validation_temp;

        %Clearing the RAM
        clear predict_validation_temp;
    end

    % Taking log on both sides of the 'Output the hypothesis' equation,
    % thus converting the product of terms to sum of terms.

    %Getting the constant_comparator, see the TrAdaBoost paper, take the
    %log of the mathematical expression on the right side of the 'Output
    %the hypothesis' equation.
    constant_comparator = ((0.5)*sum(log(beta_t(1,ceil(NumWeakClassifier/2):NumWeakClassifier).^(-1))));

    %Clearing the RAM
    clear trainX;
    clear trainY;
    clear predict_tr;
    clear predict_ts;

    %Calculating the final outcome
    for i = 1 : totalnum
        if((sum(predict_validation(i, ceil(NumWeakClassifier/2):NumWeakClassifier).*log(beta_t(1,ceil(NumWeakClassifier/2:NumWeakClassifier)).^(-1)))) >= constant_comparator)
            finalOutcome(i,1) = 1;
        end
    end

    % Return the accuracy of this round
    accuracy(jj) = sum(finalOutcome == validationY) / size(finalOutcome, 1);
    if(accuracy(jj) <= 0.50)
        random_count = random_count + 1;
        continue;
    end

    %Return the standard deviation of this round
    sd(jj) = mean(sum(abs(predict_validation - repmat(validationY, [1, NumWeakClassifier])), 2));

    %Increase the loop counter by 1 and random count by 1
    jj = jj + 1;

    %Increase the random count by 1
    random_count = random_count + 1;
end

%Display the accuracy vector and mean accuracy
accuracy
accuracy_avg = mean(accuracy)

%Display the standard deviation vector and mean standard deviation
sd
sd_avg = mean(sd)
clear;
alpha=0.1;% Set the learning rate to 0.05
epochs=250; % Set the training epochs to 100, other paramters are set to their default values
tic
%read in the yeast file;
yeast = readmatrix('yeast.csv');

%split the data into attributes and class labels
%yeast has 103 attributes and 14 classes
yeast_data = yeast(:,1:103);
yeast_target = yeast(:,104:end);
yeast_target(yeast_target == 0)=-1;
%randomly choose 67% to be train and 33% to be test
% yeast has 2417 so 33% is 800
y_cnt = size(yeast,1);
P = randperm(y_cnt);
%data should have attributes on the column
%target should have labels on the rows
ytrain_data = yeast_data(P(1:1617),:);
ytrain_target = yeast_target(P(1:1617),:)';
ytest_data = yeast_data(P(1618:end),:);
ytest_target = yeast_target(P(1618:end),:)';

%Set parameters for the BPMLL algorithm
ydim=size(ytrain_data,2);
yhidden_neuron=ceil(0.2*ydim); % Set the number of hidden neurons to 20% of the input dimensionality

% Calling the main functions
[ynets,yerrors]=BPMLL_train(ytrain_data,ytrain_target,yhidden_neuron,alpha,epochs); % Invoking the training procedure
%% 

ynet=ynets{end,1};% Set the trained neural network to the one returned after all the training epochs

[YHammingLoss,YRankingLoss,YOneError,YCoverage,YAverage_Precision,YOutputs,YThreshold,YPre_Labels]=BPMLL_test(ytrain_data,ytrain_target,ytest_data,ytest_target,ynet); % Performing the test procedure

%save data to a matlab file
toc
save('bpmll_yeast.mat');
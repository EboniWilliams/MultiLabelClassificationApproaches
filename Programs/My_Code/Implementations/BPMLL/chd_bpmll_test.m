clear;
alpha=0.05;% Set the learning rate to 0.05
epochs=100; % Set the training epochs to 100, other paramters are set to their default values
tic
% Loading the file containing the necessary inputs for calling the BPMLL
% function on the CHD dataset
chd = readmatrix('chd.csv');%removed two that had zeros

%split the data into attributes and class labels
%chd has 49 attributes and 6 classes
chd_data = chd(:,1:49);
chd_target = chd(:,50:end);
chd_target(chd_target==0)=-1;
%randomly choose 67% to be train and 33% to be test
% chd has 555 so 33% is 184
c_cnt = size(chd,1);
P2 = randperm(c_cnt);
%data should have attributes on the column
%target should have labels on the rows
train_data = chd_data(P2(1:371),:);
train_target = chd_target(P2(1:371),:)';
test_data = chd_data(P2(372:end),:);
test_target = chd_target(P2(372:end),:)';

%Set parameters for the BPMLL algorithm
dim=size(train_data,2);
hidden_neuron=ceil(0.2*dim); % Set the number of hidden neurons to 20% of the input dimensionality
% Calling the main functions
[nets,errors]=BPMLL_train(train_data,train_target,hidden_neuron,alpha,epochs); % Invoking the training procedure

net=nets{end,1};% Set the trained neural network to the one returned after all the training epochs

%Removed Ranking Loss
[HammingLoss,OneError,Coverage,Average_Precision,Outputs,Threshold,Pre_Labels]=BPMLL_test(train_data,train_target,test_data,test_target,net); % Performing the test procedure
toc
save('bpmll_CHD_100.mat');

%This is an examplar file on how the BPMLL program could be used (The main function is "BPMLL_train.m" and "BPMLL_test.m")
%
%Type 'help BPMLL_train' and 'help BPMLL_test' under Matlab prompt for more detailed information


% Loading the file containing the necessary inputs for calling the BPMLL function
load('sample data.mat'); 

%Set parameters for the BPMLL algorithm
dim=size(train_data,2);
hidden_neuron=ceil(0.2*dim); % Set the number of hidden neurons to 20% of the input dimensionality
alpha=0.05;% Set the learning rate to 0.05
epochs=100; % Set the training epochs to 100, other paramters are set to their default values

% Calling the main functions
[nets,errors]=BPMLL_train(train_data,train_target,hidden_neuron,alpha,epochs); % Invoking the training procedure

net=nets{end,1};% Set the trained neural network to the one returned after all the training epochs

[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Threshold,Pre_Labels]=BPMLL_test(train_data,train_target,test_data,test_target,net); % Performing the test procedure
toc
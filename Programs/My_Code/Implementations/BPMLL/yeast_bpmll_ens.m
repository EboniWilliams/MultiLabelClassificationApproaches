clear;
alpha=0.05;% Set the learning rate to 0.05
epochs=100; % Set the training epochs to 100, other paramters are set to their default values
%read in the yeast file;
yeast = readmatrix('yeast.csv');

%split the data into attributes and class labels
%yeast has 103 attributes and 14 classes
yeast_data = yeast(:,1:103);
yeast_target = yeast(:,104:end);
yeast_target(yeast_target == 0)=-1;
y_cnt = size(yeast,1);
%%
%  THis is the Random splits ensemble
%save the info
Hamming = zeros(1,10);
Ranking = zeros(1,10);
Error = zeros(1,10);
Cover = zeros(1,10);
Precision = zeros(1,10);
FinalNets = cell([10 1])
for ens=1:10
    %randomly choose 67% to be train and 33% to be test
    % yeast has 2417 so 33% is 800
    P = randperm(y_cnt);
    %data should have attributes on the column
    %target should have labels on the rows
    train_data = yeast_data(P(1:1617),:);
    train_target = yeast_target(P(1:1617),:)';
    test_data = yeast_data(P(1618:end),:);
    test_target = yeast_target(P(1618:end),:)';

    %Set parameters for the BPMLL algorithm
    dim=size(train_data,2);
    hidden_neuron=ceil(0.2*dim); % Set the number of hidden neurons to 20% of the input dimensionality
    % Calling the main functions
    [nets,errors]=BPMLL_train(train_data,train_target,hidden_neuron,alpha,epochs); % Invoking the training procedure
    net=nets{end,1};
    FinalNets{ens,1} = net;
    [Hamming(ens),Ranking(ens),Error(ens),Cover(ens),Precision(ens)]=BPMLL_test(train_data,train_target,test_data,test_target,net); % Performing the test procedure
end
%%

mean(Precision)
std(Precision)
save('bpmll_yeast_ens.mat');
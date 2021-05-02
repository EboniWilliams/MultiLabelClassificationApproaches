clear;
% Loading the file containing the necessary inputs for calling the BPMLL
% function on the CHD dataset
chd = readmatrix('chd.csv');

%split the data into attributes and class labels
%chd has 49 attributes and 6 classes
chd_data = chd(:,1:49);
chd_target = chd(:,50:end);
[m,n] = size(chd_target);
chd_target(chd_target == 0)=-1;%This is important
%randomly choose 67% to be train and 33% to be test
% chd has 555 so 33% is 184%removed 2 which had zeros
c_cnt = size(chd,1);
P2 = randperm(c_cnt);
%data should have attributes on the column
%target should have labels on the rows
train_data = chd_data(P2(1:371),:);
train_target = chd_target(P2(1:371),:)';
test_data = chd_data(P2(372:end),:);
test_target = chd_target(P2(372:end),:)';

%keep track of statistics
    Hamming = zeros(1,15);
    Ranking = zeros(1,15);   
    One = zeros(1,15);
    Cover = zeros(1,15);
    Precision = zeros(1,15);
    Times = zeros(1,15);

for k=1:15
    %Set parameters for the MLKNN algorithm
    Num=k;
    Smooth=1; % Set the number of nearest neighbors considered to 10 and the smoothing parameter to 1

    % Calling the main functions
    tic
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth); % Invoking the training procedure
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN); % Performing the test procedure
    %saving the data
    Times(k) = toc;
    Hamming(k) = HammingLoss;
    Ranking(k) = RankingLoss;
    One(k) = OneError;
    Cover(k) = Coverage;
    Precision(k) = Average_Precision;
end
save('mlknn_CHD.mat');


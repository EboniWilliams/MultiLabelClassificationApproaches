
%read in the yeast file;
yeast = readmatrix('yeast.csv');

%split the data into attributes and class labels
%yeast has 103 attributes and 14 classes
yeast_data = yeast(:,1:103);
yeast_target = yeast(:,104:end);
yeast_target(yeast_target == 0)=-1;%This is important
%randomly choose 67% to be train and 33% to be test
% yeast has 2417 so 33% is 800
y_cnt = size(yeast,1);
P = randperm(y_cnt);
%data should have attributes on the column
%target should have labels on the rows
train_data = yeast_data(P(1:1617),:);
train_target = yeast_target(P(1:1617),:)';
test_data = yeast_data(P(1618:end),:);
test_target = yeast_target(P(1618:end),:)';

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
save('mlknn_yeast.mat');

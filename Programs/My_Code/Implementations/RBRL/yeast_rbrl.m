clear;
%% read the dataset
% the dataset for the kernel model has been normalized with 0 mean and 1
% std. % Did I do this
%read in the yeast file;
yeast = readmatrix('yeast.csv');

%split the data into attributes and class labels
%yeast has 103 attributes and 14 classes
yeast_data = normalize(yeast(:,1:103));
yeast_target = yeast(:,104:end);
yeast_target(yeast_target == 0)=-1;%This is important
%randomly choose 67% to be train and 33% to be test
% yeast has 2417 so 33% is 800
y_cnt = size(yeast,1);
P = randperm(y_cnt);
%data should have attributes on the column
%target should have labels on the rows
X_train = yeast_data(P(1:1617),:);
Y_train = yeast_target(P(1:1617),:);
X_test = yeast_data(P(1618:end),:);
Y_test = yeast_target(P(1618:end),:);
% append another feature (all equals to 1) to the data
num_feature_origin = size(X_train, 2);
X_train(:, num_feature_origin + 1) = 1;
X_test(:, num_feature_origin + 1) = 1;

%% train the model in the train dataset and predict in the test dataset 
NIter = 2000;
% set the RBF kernel hyper-parameter
sigma = 1 / num_feature_origin; 
lambda1 = 1;
lambda2 = 0.01;
lambda3 = 0.1;
tic
% % For RBF kernel model
[ A, obj ] = train_kernel_RBRL_APG( X_train, Y_train, lambda1, lambda2, lambda3, sigma, NIter);
[ pre_Label_test, pre_F_test ] = Kernel_Predict( X_train, X_test, A, sigma );
%% For linear model
% [ W, obj ] = train_linear_RBRL_APG( X_train, Y_train, lambda1, lambda2, lambda3, NIter );
% [ pre_Label_test, pre_F_test ] = Predict( +X_test, W );

% Calling the main functions
%% evaluate the performance of the model
[ test_HammingLoss, test_SubsetAccuracy, test_AccuracyExample, test_F1_Micro_Label, test_F1_Macro_Label, test_F1_Example, ...
    test_Ranking_Loss, test_Coverage, test_Average_Precision, test_One_Error, test_AUC_Macro_Label ] = Evaluation_Metrics( pre_Label_test, pre_F_test, Y_test );
plot(obj);
%saving the data
time = toc
save('rbrl_yeast_norm_kernel.mat');
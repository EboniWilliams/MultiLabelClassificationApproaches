clear;
% Loading the file containing the necessary inputs for calling the RBRL
% function on the CHD dataset
chd = readmatrix('chd2.csv');

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
X_train = chd_data(P2(1:371),:);
Y_train = chd_target(P2(1:371),:);
X_test = chd_data(P2(372:end),:);
Y_test = chd_target(P2(372:end),:);
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
%% evaluate the performance of the model
[ test_HammingLoss, test_SubsetAccuracy, test_AccuracyExample, test_F1_Micro_Label, test_F1_Macro_Label, test_F1_Example, ...
    test_Ranking_Loss, test_Coverage, test_Average_Precision, test_One_Error, test_AUC_Macro_Label ] = Evaluation_Metrics( pre_Label_test, pre_F_test, Y_test );
plot(obj);
%saving the data
time = toc
save('rbrl_chd_unnorm_kernel.mat');
%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% clear;
% % Loading the file containing the necessary inputs for calling the RBRL
% % function on the CHD dataset
% chd = readmatrix('chd2.csv');
% 
% %split the data into attributes and class labels
% %chd has 49 attributes and 6 classes
% chd_data = chd(:,1:49);
% chd_target = chd(:,50:end);
% [m,n] = size(chd_target);
% chd_target(chd_target == 0)=-1;%This is important
% %randomly choose 67% to be train and 33% to be test
% % chd has 555 so 33% is 184%removed 2 which had zeros
% c_cnt = size(chd,1);
% P2 = randperm(c_cnt);
% %data should have attributes on the column
% %target should have labels on the rows
% X_train = chd_data(P2(1:371),:);
% Y_train = chd_target(P2(1:371),:);
% X_test = chd_data(P2(372:end),:);
% Y_test = chd_target(P2(372:end),:);
% % append another feature (all equals to 1) to the data
% num_feature_origin = size(X_train, 2);
% X_train(:, num_feature_origin + 1) = 1;
% X_test(:, num_feature_origin + 1) = 1;
% %% train the model in the train dataset and predict in the test dataset 
% NIter = 2000;
% % set the RBF kernel hyper-parameter
% sigma = 1 / num_feature_origin; 
% lambda1 = 1;
% lambda2 = 0.01;
% lambda3 = 0.1;
% tic
% %% For linear model
% [ W, obj ] = train_linear_RBRL_APG( X_train, Y_train, lambda1, lambda2, lambda3, NIter );
% [ pre_Label_test, pre_F_test ] = Predict( X_test, W );
% 
% % Calling the main functions
% %% evaluate the performance of the model
% [ test_HammingLoss, test_SubsetAccuracy, test_AccuracyExample, test_F1_Micro_Label, test_F1_Macro_Label, test_F1_Example, ...
%     test_Ranking_Loss, test_Coverage, test_Average_Precision, test_One_Error, test_AUC_Macro_Label ] = Evaluation_Metrics( pre_Label_test, pre_F_test, Y_test );
% plot(obj);
% %saving the data
% time = toc
% save('rbrl_chd_unnorm_linear.mat');
%% 
%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
clear;
% Loading the file containing the necessary inputs for calling the RBRL
% function on the CHD dataset
chd = readmatrix('chd2.csv');

%split the data into attributes and class labels
%chd has 49 attributes and 6 classes
chd_data = normalize(chd(:,1:49));
chd_target = chd(:,50:end);
chd_target(chd_target == 0)=-1;%This is important
%randomly choose 67% to be train and 33% to be test
% chd has 555 so 33% is 184%removed 2 which had zeros
c_cnt = size(chd,1);
P2 = randperm(c_cnt);
%data should have attributes on the column
%target should have labels on the rows
X_train = chd_data(P2(1:371),:);
Y_train = chd_target(P2(1:371),:);
X_test = chd_data(P2(372:end),:);
Y_test = chd_target(P2(372:end),:);
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
%% evaluate the performance of the model
[ test_HammingLoss, test_SubsetAccuracy, test_AccuracyExample, test_F1_Micro_Label, test_F1_Macro_Label, test_F1_Example, ...
    test_Ranking_Loss, test_Coverage, test_Average_Precision, test_One_Error, test_AUC_Macro_Label ] = Evaluation_Metrics( pre_Label_test, pre_F_test, Y_test );
plot(obj);
%saving the data
time = toc
save('rbrl_chd_norm_kernel.mat');
%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
clear;
% Loading the file containing the necessary inputs for calling the RBRL
% function on the CHD dataset
chd = readmatrix('chd2.csv');

%split the data into attributes and class labels
%chd has 49 attributes and 6 classes
chd_data = normalize(chd(:,1:49));
chd_target = chd(:,50:end);
[m,n] = size(chd_target);
chd_target(chd_target == 0)=-1;%This is important
%randomly choose 67% to be train and 33% to be test
% chd has 555 so 33% is 184%removed 2 which had zeros
c_cnt = size(chd,1);
P2 = randperm(c_cnt);
%data should have attributes on the column
%target should have labels on the rows
X_train = chd_data(P2(1:371),:);
Y_train = chd_target(P2(1:371),:);
X_test = chd_data(P2(372:end),:);
Y_test = chd_target(P2(372:end),:);
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
%% For linear model
[ W, obj ] = train_linear_RBRL_APG( X_train, Y_train, lambda1, lambda2, lambda3, NIter );
[ pre_Label_test, pre_F_test ] = Predict( X_test, W );

Calling the main functions
%% evaluate the performance of the model
[ test_HammingLoss, test_SubsetAccuracy, test_AccuracyExample, test_F1_Micro_Label, test_F1_Macro_Label, test_F1_Example, ...
    test_Ranking_Loss, test_Coverage, test_Average_Precision, test_One_Error, test_AUC_Macro_Label ] = Evaluation_Metrics( pre_Label_test, pre_F_test, Y_test );
plot(obj);
%saving the data
time = toc
save('rbrl_chd_norm_linear.mat');


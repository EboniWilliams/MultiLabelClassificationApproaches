
%This method doesn't work currently. It produces NaN and complex numbers. I
%haven't figured out why yet.
clear;
% Loading the file containing the necessary inputs for calling the BPMLL
% function on the CHD dataset
chd = readmatrix('chd.csv');

%split the data into attributes and class labels
%chd has 49 attributes and 6 classes
num_attr = 49;
chd_data = chd(:,1:num_attr);
chd_target = chd(:,num_attr+1:end);
[num_data,num_lbls] = size(chd_target); 
chd_target(chd_target == 0)=-1;%This is important
%randomly choose 67% to be train and 33% to be test
% chd has 555 so 33% is 184%removed 2 which had zeros
num_train = 371;
c_cnt = size(chd,1);
P = randperm(c_cnt);
%data should have attributes on the column
%target should have labels on the rows
train_data = chd_data(P(1:num_train),:);
train_target = chd_target(P(1:num_train),:)';
test_data = chd_data(P(num_train+1:end),:);
test_target = chd_target(P(num_train+1:end),:)';
%Set parameters for the MLKNN algorithm
k=10;
Smooth=1; % Set the number of nearest neighbors considered to 10 and the smoothing parameter to 1
% Find the weights on the training data
W = brwd_weights(train_data,train_target,8,0.1,k,100)
% Calling the main functions
% For those attempting to pick up where I left off, this method needs to be
% modified to incorporate the weights in the label classifications. I
% didn't get to it because I spent so much time trying to fix the weights
% function.
% [Prior,PriorN,Cond,CondN]=MLKNN_modified_train(train_data,train_target,k,Smooth); % Invoking the training procedure
%%
%asymmetric distance
%%
function dW = asym_dist(q,x,w)
    dW =sqrt(sum(q-x,2))/w;
end
% the sigmoid function
function S = Sig(r,bet)
    S = 1/(1+exp(bet*(1-r)));
end
%derivative of the sigmoid function
function dS =  dSig(r,bet)
    dS = bet*Sig(r,bet)*(1-Sig(r,bet));
end

function r_w = rr(A,B,C,w_A,w_C,ep)
    if nargin == 3
        ep = 0;
    end
    r_w = asym_dist(A,B,w_A)/(asym_dist(A,C,w_C)+ep);
end

function W = brwd_weights(XX,YY,bet,eta,k,max_iters)
    [N,L] = size(YY);
    D = size(XX,1);
    %Computing distance between instances
    %using same method as in MLKNN
    mat1=concur(sum(XX.^2,2),D);
    mat2=mat1';
    dist_matrix=mat1+mat2-2*XX*XX';
    dist_matrix=sqrt(dist_matrix);
    for i=1:D
        dist_matrix(i,i)=realmax;
    end
    
    W = ones(L,N);
    for ll=1:L
        ii=0;
        while ii < max_iters
                % find the the true/false positive/negatives for the label
                tp = 0;
                fn = 0;
                fp = 0;
                for proto = find(YY(ll,:) == 1) % for each positive instance
                    % k/2 closest friends
                    [~,index] = sort(dist_matrix(proto,YY(ll,:) == 1));
                    Nf = index(1:ceil(k/2));
                    % k/2 closest enemies
                    [~,index] = sort(dist_matrix(proto,YY(ll,:) == -1));
                    Ne = index(1:ceil(k/2));
                
                    %kth nearest neighbor of proto for r
                   [~,index] = sort(dist_matrix(proto,:));
                    for friend = Nf
                        rval = rr(XX(proto,:),XX(friend,:),XX(index(k),:),W(proto,ll),W(index(k),ll),0.01);
                        tp = tp + Sig(rval^(-1),bet);
                        fn = fn + Sig(rval,bet);
                    end
                    for enemy = Ne
                        rval = rr(XX(proto,:),XX(enemy,:),XX(index(k),:),W(proto,ll),W(index(k),ll),0.01);
                        tp = tp + Sig(rval,bet);
                        fn = fn + Sig(rval^(-1),bet);
                    end
                end
                tp = tp/(2*ceil(k/2));
                fn = fn/(2*ceil(k/2));
                for proto = find(YY(ll,:) == -1) % for each negative instance
                    % k/2 closest friends
                    [~,index] = sort(dist_matrix(proto,YY(ll,:) == 1));
                    Nf = index(1:ceil(k/2));
                    % k/2 closest enemies
                    [~,index] = sort(dist_matrix(proto,YY(ll,:) == -1));
                    Ne = index(1:ceil(k/2));   
                    %kth nearest neighbor of x for r
                    [~,index] = sort(dist_matrix(proto,:));
                    for friend = Nf
                        rval = rr(XX(proto,:),XX(friend,:),XX(index(k),:),W(proto,ll),W(index(k),ll),0.01);
                        fp = fp + Sig(rval,bet);
                    end
                    for enemy = Ne
                        rval = rr(XX(proto,:),XX(enemy,:),XX(index(k),:), W(proto,ll),W(index(k),ll),0.01);
                        fp = fp + Sig(rval^(-1),bet);
                    end
                end
                fp = fp/(2*ceil(k/2));  
            for jj=1:N
                label = YY(ll,jj);
                % find the change
                dtp = 0;
                dfn = 0;
                dfp = 0;
                for proto = find(YY(ll,:) == 1) % for each positive instance     
                    % if x is in the nearest friends
                    [~,index] = sort(dist_matrix(proto,YY(ll,:) == 1));
                    [~,temp] = sort(dist_matrix(proto,:));
                    if ismember(jj,index(1:ceil(k/2)))
                       rp = rr(XX(proto,:),XX(jj,:),XX(temp(k),:),W(proto,ll),W(temp(k),ll),0.01);
                       dtp = dtp + label*(dSig(rp,bet)^(-label))*(rp^(-label))/W(jj,ll);
                       dfn = dfn + -label*(dSig(rp,bet)^(label))*(rp^(label))/W(jj,ll);
                    end
                end
                % for each negative instance
                for proto = find(YY(ll,:) == -1) 
                    % if x is in the nearest friends
                    [~,index] = sort(dist_matrix(proto,YY(ll,:) == 1));
                    [~,temp] = sort(dist_matrix(proto,:));
                    if ismember(jj,index(1:ceil(k/2)))
                       rn = rr(XX(proto,:),XX(jj,:),XX(temp(k),:),W(proto,ll),W(temp(k),ll),0.01);
                       dfp = dfp + label*(dSig(rn,bet)^(-label))*(rn^(-label))/W(jj,ll);
                    end
                end
                dF = (2/L)*(dtp*fp-dfp*tp+dfp*fn-dfn*tp)/(2*tp+fp+fn)^2;
                %update the weights
                W(jj,ll) = W(jj,ll) + eta*dF;
            end
        end
    end
end
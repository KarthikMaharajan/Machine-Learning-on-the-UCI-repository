%% Breast Cancer Wisconsin

clc
clear all
close all

%% importing data

data = importfile('data.txt');

X = data(:,2:10);
L = data(:,11);

n = size(X,1);

%
idx = randperm(n);

X = X(idx,:);
L = L(idx,:);

%%  Algorithm: Support Vector Machine: Testset with K_fold cross validation

P = [0.1, 0.2, 0.3, 0.4, 0.5];

ER_test = [];
ER_training = [];

for i = 1:5
    
    n_train = floor(P(i)*n);
    
    Xtrain = X(idx(1:n_train),:);
    Ltrain = L(idx(1:n_train));
    
    Xtest = X(idx(n_train+1:end),:);
    Ltest = L(idx(n_train+1:end));
    
    % Cross Validation for choosing C
    C = logspace(-3,1,40);
    
    for c = 1:40
        
        % SVM command, one versus all
        t = templateSVM('Standardize',1,'BoxConstraint',C(c), 'Solver','SMO');
        CVMdl = fitcecoc(Xtrain,Ltrain,'Learners',t,'Coding','onevsall','KFold',10,'Verbose',0);
        
        % computing trainingset error
        classLoss = kfoldLoss(CVMdl);
        
        figure(i)
        semilogx(C(c),classLoss,'b*')
        title('Training set error vs. C ');
        xlabel('C');
        ylabel('Error');
        hold on
    end
    
    
    % SVM command, one versus all, choose C=0.01 by 10-Fold Cross Validation
    t = templateSVM('Standardize',1,'BoxConstraint',0.01, 'Solver','SMO');
    Mdl = fitcecoc(Xtrain,Ltrain,'Learners',t,'Coding','onevsall','Verbose',0);
    
    % computing trainingset error
    classLoss = resubLoss(Mdl);
    
    % computing testset error
    error = loss(Mdl,Xtest,Ltest);
    
    % creating confusion matrix
    Ltest_hat = predict(Mdl,Xtest);
    [D,Order] = confusionmat(Ltest,Ltest_hat);
    
    ER_test    = [ER_test;error];
    
    ER_training = [ER_training;classLoss];
end

% matrix of errors
ER_training

ER_test

%%  Algorithm: K_means

% number of clusters
k = 2;

% k-Means command
[idx,C] = kmeans(X,k,'Display','off','Distance','sqeuclidean','OnlinePhase','on','Replicates',5);

LCluster = [];

% labeling each cluster
for i = 1:k
    
    d           = pdist2(X,C(i,:),'euclidean'); % the distance between data to Cluster centroid locations
    [~,ind_min] = min(d);
    LCluster(i) = L(ind_min);
    
end

Lhat  = LCluster(idx)';

% computing error and confusion matrix
kmeans_error = mean(abs(Lhat - L)>0)
[D,Order]    = confusionmat(L,Lhat)

% script.m
% Author: Joseph K Aicher
% Demonstrates how to evaluate with cross-validation, train, and test our
% Naive Bayes model

%% Set up training environment
% load our data
load('train.mat');  % has X_train_bag, Y_train, 
N = size(X_train_bag, 1);
P = size(X_train_bag, 2);

% set up cross-validation
K = 10;
rng(1776);  % random seed for folds
cv_ndx = crossvalind('Kfold', N, K);

%% Set up function to train our Naive Bayes model
% no preprocessing for this most vanilla of Naive Bayes models...
preprocess = @(pseudocount) (@(X, Y, W) preprocess_bernoulliNB(X, Y, W, 1));
% we will make predictions with CPE model
predict = @(pseudocount) chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess(pseudocount));
% given cost, predict(cost) is a function of X, Y, W, which will output [f, Yt], where
% f makes prediction on new input data, and Yt is our predicted labels on the training data X

%% Do cross-validation on cost
pseudocounts = [0.1, 0.5, 1, 5];
for pseudocount = pseudocounts
    [NB_mean, NB_sd] = evaluate_predict(predict(pseudocounts), X_train_bag, Y_train, cv_ndx, true);
    fprintf('error: %f +/- %f, pseudocount: %f\n', NB_mean, NB_sd, pseudocount);
end

%% Train on cost
% we honestly arbitrarily chose pseudocount=1 and retconned the cross-validation above
make_prediction = predict(1);
[f, Yt] = make_prediction(X_train_bag, Y_train, ones(size(Y_train)));
% f is our function to make predictions
save NB.mat f;  % save it to NB.mat so that predict_labels can use it too

%% Make predictions
newX = X_train_bag;  % change for different predictions
newY = Y_train;  % change for different true labels
Y_est = f(newX);  % new predictions
fprintf('Cost %f of predictions on newX\n', performance_measure(Y_est, newY));
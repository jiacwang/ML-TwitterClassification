% script.m
% Author: Joseph K Aicher
% Demonstrates how to evaluate with cross-validation, train, and test our
% logistic regression model

%% Set up training environment
% load our data
load('train.mat');  % has X_train_bag, Y_train, 
N = size(X_train_bag, 1);
P = size(X_train_bag, 2);

% set up cross-validation
K = 10;
rng(1776);  % random seed for folds
cv_ndx = crossvalind('Kfold', N, K);

%% Set up function to train our preferred logistic regression model
% stem vocabulary -- from 10k features to ~7k features
stem_vocabulary = @(X, Y, W) preprocess_stem_vocabulary(X, Y, W);
% logistic for given cost, makes soft probability output
logistic = @(cost) (@(X, Y, W) preprocess_logistic(X, Y, W, cost, true, false));
% together these are our preprocessing steps
preprocess = @(cost) chain_preprocess(stem_vocabulary, logistic(cost));
% we will make predictions with CPE model
predict = @(cost) chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess(cost));
% given cost, predict(cost) is a function of X, Y, W, which will output [f, Yt], where
% f makes prediction on new input data, and Yt is our predicted labels on the training data X

%% Do cross-validation on cost
costs = [0.1, 0.3, 0.5, 1];
for cost = costs
    [logistic_mean, logistic_sd] = evaluate_predict(predict(cost), X_train_bag, Y_train, cv_ndx, true);
    fprintf('error: %f +/- %f, cost: %f\n', logistic_mean, logistic_sd, cost);
end

%% Train on cost
% we really liked cost = 0.3...
make_prediction = predict(0.3);
[f, Yt] = make_prediction(X_train_bag, Y_train, ones(size(Y_train)));
% f is our function to make predictions
save submission3.mat f;  % save it to submission3.mat so that predict_labels can use it too

%% Make predictions
newX = X_train_bag;  % change for different predictions
newY = Y_train;  % change for different true labels
Y_est = f(newX);  % new predictions
fprintf('Cost %f of predictions on newX\n', performance_measure(Y_est, newY));
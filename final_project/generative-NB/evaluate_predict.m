function [ mean_error, std_error ] = evaluate_predict( predict, X, Y, cv_ndx, verbose, first_folds )
%EVALUATE_PREDICT evaluates cross-validation error of predict using Xtrain,
%Ytrain with the folds specified by cv_ndx
%   predict: @(X, Y, W) --> [f, Yt], which we can use f on the held out set
%   X is NxP matrix of predictors
%   Y is Nx1 matrix of labels
%   cv_ndx is Nx1 matrix of fold assignments range 1...K
%   verbose if specified says if we want to print output as we go

if nargin < 5
    verbose = false;
end

K = max(cv_ndx);

if nargin < 6
    first_folds = K;
else
    first_folds = max(1, min(first_folds, K));
end

cv_error = zeros(first_folds, 1);  % initialize place for errors
if verbose
    fprintf("Beginning cross-validation\n");
end
for k=1:first_folds  % loop over each of the folds
    ndx = cv_ndx == k;  % logical where trues correspond to test fold
    % train the model
    Xtrain = X(~ndx, :);
    Ytrain = Y(~ndx, :);
    weights = ones(size(Ytrain));
    [f, ~] = predict(Xtrain, Ytrain, weights);
    % evaluate the model on the held out set
    Xtest = X(ndx, :);
    Ytest = Y(ndx, :);
    Ypred = f(Xtest);
    cv_error(k) = performance_measure(Ypred, Ytest);
    if verbose
        fprintf("Fold %d, Error %f\n", k, cv_error(k));
    end
end
mean_error = mean(cv_error);
std_error = std(cv_error);
end


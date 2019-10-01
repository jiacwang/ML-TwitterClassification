function [mean_error, std_error] = evaluate_method(X, Y, cv_ndx, run_model, verbose)
% X is NxP matrix of predictors
% Y is Nx1 matrix of labels
% cv_ndx is Nx1 matrix of fold assignments range 1...K
% run_model: @(Xtrain, Ytrain, Xtest) obtains predicted labels

if nargin < 5
    verbose = false;
end

K = max(cv_ndx);  % get K from cv_ndx
cv_error = zeros(K, 1);  % initialize place for errors
if verbose
    fprintf("Beginning cross-validation\n");
end
for k=1:K  % loop over each of the folds
    ndx = cv_ndx == k;  % logical where trues correspond to test fold
    predictions = run_model(X(~ndx,:), Y(~ndx,:), X(ndx,:));
    cv_error(k) = performance_measure(predictions, Y(ndx,:));
    if verbose
        fprintf("Fold %d, Error %f\n", k, cv_error(k));
    end
end
mean_error = mean(cv_error);
std_error = std(cv_error);
end
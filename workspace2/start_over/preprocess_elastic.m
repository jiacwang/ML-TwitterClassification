function [ f, Xt ] = preprocess_elastic( X, Y, W, alpha, lambda, standardize )
%PREPROCESS_elastic Computes probabilities for possible classes using
%elastic net of binomial model
%   Detailed explanation goes here

N = size(X, 1);
P = size(X, 2);

weights = W ./ sum(W);  % their weights normalize to 1

% initialize fits for each
beta = zeros(P, 5);
intercept = zeros(1, 5);

% loop over one vs rest fits
for k=1:5
    [B, fit_info] = lassoglm(X, Y == k, 'binomial', 'Alpha', alpha, 'Lambda', lambda, 'Weights', weights, 'Standardize', standardize);
    % extract beta and intercept here
    beta(:, k) = B(:, 1);
    intercept(k) = fit_info.Intercept;
end

% create tranasform function
f = @(inputX) preprocess_logistic_helper(inputX, beta, intercept);
Xt = f(X);


end

function Xt = preprocess_logistic_helper(X, beta, intercept)

Xt = 1./(1 + exp(intercept + X * beta));  % gets probability space estimate

end


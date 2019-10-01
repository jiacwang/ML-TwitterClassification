function [ f, Xt ] = lassoglm_scores( X, Y, alpha, probabilities, NB_features)
%lassoglm_scores Estimates per class binomial models of glmnet
%regularization path on lambda for given alpha, with 5-fold
%cross-validation to pick lambda. Used to transform X to log-odds scores
%(or probability)
%   X input matrix
%   Y labels
%   alpha mixing between l1 and l2 penalties
%   probabilities is boolean to indicate whether or not to exponentiate to
%   proportional to probability
%   NB_features boolean indicating whether to multiply by NB inspired
%   weight

if nargin < 5
    NB_features = false;
end

% constant costs matrix for weighting regressions
costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

% initialize fits for each...
best_fit_beta = zeros(size(X, 2), 5);
best_fit_intercept = zeros(1, 5);

for k=1:5
    % what weights do we want to give?
    weights = costs(Y, k);  % cost of predicting k for this example
    weights(Y==k) = max(costs(:, k));  % replace zeros with maximum cost
    weights = weights / sum(weights);  % normalize to have unit sum
    if NB_features
        feature_weights = calculate_NB_weights(X, Y, k);
        Xk = X .* feature_weights;
    else
        Xk = X;
    end
    [B, fit_info] = lassoglm(Xk, Y == k, 'binomial', 'Alpha', alpha, 'CV', 10, 'NumLambda', 5, 'Weights', weights, 'Standardize', false);
    best_fit_beta(:, k) = B(:, fit_info.IndexMinDeviance);
    best_fit_intercept(k) = fit_info.Intercept(fit_info.IndexMinDeviance);
end

f = @(inputX) lassoglm_scores_helper(inputX, best_fit_beta, best_fit_intercept, probabilities);
Xt = f(X);


end

function Xt = lassoglm_scores_helper(X, beta, intercept, probabilities)
scores = X*beta;
scores = scores + intercept;
if probabilities
    Xt = exp(scores);
else
    Xt = scores;
end
end

function feature_weights = calculate_NB_weights(X, Y, k)
% calculate weights for features of X using X, Y, for k-th label
% wt = 1/(alpha - log(max(beta,P(xj|y=k)))
alpha = 0.1;
beta = 0.01;
Xk = X(Y==k,:);
Nk = shape(Xk, 1);
empirical_p = sum(Xk > 0) / Nk;
feature_weights = 1 ./ (alpha - log(max(beta, empirical_p)));
end


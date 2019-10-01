function [ f, Xt ] = preprocess_logisticNB( X, Y, W, cost, pseudocount)
%PREPROCESS_LOGISTIC Uses liblinear ridge-penalized logistic regression to
%fit K (eg 5) separate models to obtain probability estimates
%   X: N x P input matrix
%   Y: N x 1 labels
%   W: N x 1 weights
%   cost: cost penalty for liblinear
%   pseudocount: pseudocount for NB weights

% unfortunately liblinear doesn't have a great way of scaling individual
% observations weights...

N = size(X, 1);
P = size(X, 2);

% transformations for input data X here before liblinear...
transformations = cell(1, 5);

% coefficients
beta = zeros(P+1, 5);

% loop over the classes for classification
for k = 1:5
    [fk, Xk] = preprocess_1vall_NBfeatures(X, Y, W, k, pseudocount, true);
    transformations{k} = fk;
    % set up liblinear_options
    liblinear_options = sprintf('-s 0 -q -c %f', cost);
    % fit the model
    modelk = liblinear_train(Y, Xk, liblinear_options);
    % use the result
    beta(:, k) = modelk.w(find(modelk.Label == k, 1),:)';
end

% get overall transformation...
f = @(inputX) preprocess_logistic_helper(inputX, beta, transformations);
% apply it to original data
Xt = f(X);

end

function Xt = preprocess_logistic_helper(X, beta, transformations)
% each of the beta were trained agains others... so don't normalize scores
% against each other...

N = size(X, 1);
K = length(transformations);

Xt = zeros(N, K);
for k=1:K
    Xt(:, k) = 1 ./ (1 + exp(-transformations{k}(X) * beta(:, k)));
end
end
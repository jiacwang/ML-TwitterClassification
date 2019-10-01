function [ f, Xt ] = preprocess_logistic( X, Y, W, cost, NB_features, weight_labels )
%PREPROCESS_LOGISTIC Uses liblinear ridge-penalized logistic regression to
%fit K (eg 5) separate models to obtain probability estimates
%   X: N x P input matrix
%   Y: N x 1 labels
%   W: N x 1 weights
%   cost: cost penalty for liblinear
%   NB_features: if true, uses preprocess_1vall_NBfeatures to reweight
%   features according to NB weights. If number, used as pseudocount
%   parameter, too
%   weight_labels: if true, scales cost for different labels to reflect
%   cost-sensitive...

% unfortunately liblinear doesn't have a great way of scaling individual
% observations weights...

N = size(X, 1);
P = size(X, 2);

% get costs if asked for
if weight_labels == true
    % we should do some weighting here, though...
    weightN = sum(W);
    Yvec = bsxfun(@eq, Y(:), 1:5);  % vectorize Y (1 x K)
    Py = sum(W .* Yvec) / weightN;  % prior on labels (1 x K)
    % costs matrix is...
    costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    % this is great, but we want to weight self to -- use average
    % misclassification error using prior probability
    costs_diag = (costs * Py')./((1-eye(5))* Py');
    costs(logical(eye(5))) = costs_diag;
    % so costs is a 5x5 matrix, where the k-th row gives us the per-class
    % costs we want to use when classifying...
end

% function to add bias vector to X
add_bias = @(inputX) [inputX, ones(size(inputX, 1), 1)];

% transformations for input data X here before liblinear...
transformations = cell(1, 5);

% coefficients
beta = zeros(P+1, 5);

% loop over the classes for classification
for k = 1:5
    if NB_features
        [fk, X_nbk] = preprocess_1vall_NBfeatures(X, Y, W, k, NB_features);
        transformations{k} = @(inputX) add_bias(fk(inputX));
        Xk = add_bias(X_nbk);
    else
        transformations{k} = add_bias;
        Xk = add_bias(X);
    end
    % set up liblinear_options
    liblinear_options = sprintf('-s 0 -q -c %f', cost);
    % add weights if relevant
    if weight_labels
        cost_options = '';
        for kp = 1:5
            cost_options = sprintf(' %s -w%d %f', cost_options, kp, costs(k, kp));
        end
        liblinear_options = sprintf('%s%s', liblinear_options, cost_options);
    end
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
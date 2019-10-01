function [ f, Xt ] = preprocess_logisticEM( Xo, Y, ~, cost, pseudocount, Xvo, unobs_weight)
%PREPROCESS_LOGISTICEM trains logistic using EM on all data we have
%   Xo: N x P input matrix
%   Y: N x 1 labels
%   W: N x 1 weights
%   cost: cost penalty for liblinear train
%   pseudocount: pseudocount for NB features
%   Xvo: unmeasured data to use for training
%   unobs_weight: how to weight unobserved data against observed (< 1)


% hard-coded stemming of X
[stem, X] = preprocess_stem_vocabulary(Xo, [], []);
Xv = stem(Xvo);

% initialize our empty hidden observations (E-step)
Yv = zeros(size(Xv, 1), 5);  % hidden variables from expectation (we set them to zero to start)...

% train our parameters (EM iteration starting with M-step)
train_eps = 0.001;
train_iterations = 5;
[beta, transformations] = iterate_maximization_expectation(X, Y, Xv, Yv, cost, pseudocount, unobs_weight, train_eps, train_iterations);

% get our transformation function
f = @(inputX) preprocess_logisticEM_helper(inputX, beta, transformations, stem, X, Y, Xv, cost, pseudocount, unobs_weight);
% transform our original data
Xt = f(Xo);



end

function Xt = preprocess_logisticEM_helper(Xo, beta, transformations, stem, X_obs, Y_obs, Xv, cost, pseudocount, unobs_weight)
% uses expectation maximization to update beta, transformations using input
% X and make our best prediction of the probability it is one of the few
% classes...
% Xo is unprocessed input

X = stem(Xo);  % get fully preprocessed X...
% is this the validation data? get appropriate set of all unobserved data
if isequal(X, Xv)
    Xu = X;  % same as validation, so only one time
else
    Xu = [X; Xv];  % not validation, so we can use both as unobserved...
end

% first expectation step, get Yu_0
Yu_0 = logisticEM_expectation(Xu, beta, transformations);

% perform iteration to get final beta, transformations
predict_eps = 0.001;
predict_iterations = 5;
[opt_beta, opt_transformations] = iterate_maximization_expectation(X_obs, Y_obs, Xu, Yu_0, cost, pseudocount, unobs_weight, predict_eps, predict_iterations);
% make our final predictions...
Xt = logisticEM_expectation(X, opt_beta, opt_transformations);


end

function [beta, transformations] = iterate_maximization_expectation(X, Y, Xu, Yu_0, cost, pseudocount, unobs_weight, eps, max_iter)
% does iteration of EM, starting with M-step, to obtain converged
% beta/transformations
% eps: l-infinity criterion for convergence, on Yu_t
% max_iter: maximum number of iterations

% loop this
% initialize this
Yu_t = Yu_0;
for iter = 1:max_iter
    % maximization step, get new beta and transformations
    [beta_t, transformations_t] = logisticEM_maximization(X, Y, Xu, Yu_t, cost, pseudocount, unobs_weight);
    % expectation step, get new Yu
    Yu_new = logisticEM_expectation(Xu, beta_t, transformations_t);
    % check for convergence
    dYu_inf = mean(max(abs(Yu_new - Yu_t), [], 2));
    fprintf("Iteration %d, dYu_inf %f\n", iter, dYu_inf);
    if dYu_inf < eps
        fprintf("Converged in %d iterations\n", iter);
        % converged
        break;
    else
        % update Yu_t and repeat
        Yu_t = Yu_new;
    end
end

% either converged, or run out of iterations, return our final values
beta = beta_t;
transformations = transformations_t;


end

function Yu = logisticEM_expectation(Xu, beta, transformations)
% computes current expectation of labels of Xu...
N = size(Xu, 1);
K = length(transformations);

Yu = zeros(N, K);
for k = 1:K
    Yu(:, k) = 1 ./ (1 + exp(-transformations{k}(Xu) * beta(:, k)));
end

% normalize output...
Yu = Yu ./ sum(Yu, 2);  % sum of rows is equal to 1
end

function [beta, transformations] = logisticEM_maximization(X, Y, Xu, Yu, cost, pseudocount, unobs_weight)
% obtains new beta, transformations by training...

% get number of parameters
P = size(X, 2);

% initialize return values
transformations = cell(1, 5);
beta = zeros(P + 1, 5);

% normalize Yu to the unobs_weight
Yu = unobs_weight .* Yu;

% loop over the classes for training separate classifiers
for k = 1:5
    [fk, Xk] = current_NB_features(X, Y, Xu, Yu, k, pseudocount);
    transformations{k} = fk;
    % set up liblinear options
    liblinear_options = sprintf('-s 0 -q -c %f', cost);
    % fit the model
    modelk = liblinear_train(Y, Xk, liblinear_options);
    % use the result
    beta(:, k) = modelk.w(find(modelk.Label == k, 1), :)';
end

end

function [f, Xt] = current_NB_features(X, Y, Xu, Yu, k, pseudocount)
% creates log_ratios of NB for k vs rest classification


% count appearances of features for k vs not k
Nj_k = sum(1 .* (X(Y == k,:) ~= 0)) + sum(Yu(:, k) .* (Xu ~= 0)) + pseudocount;
Nj_notk = sum(1 .* (X(Y ~= k,:) ~= 0)) + sum((1- Yu(:, k) .* (Xu ~= 0))) + pseudocount;
% get log-ratio between these
log_ratio = (log(Nj_k) - log(sum(Nj_k))) - (log(Nj_notk) - log(sum(Nj_notk)));
% get bias value
bias = log(sum(Y == k) + sum(Yu(:, k))) - log(sum(Y ~= k) + sum(1-Yu(:, k)));

% get transformation
f = @(inputX) [inputX .* log_ratio, bias .* ones(size(inputX, 1), 1)];
Xt = f(X);

end


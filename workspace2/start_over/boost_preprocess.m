function new_preprocess = boost_preprocess( predict, n_estimators )
%BOOST_PREPROCESS Uses SAMME boosting with the predict learner
%   predict: @(X, Y, W) --> [f, Yt] predictor
%   n_estimators: how many weighted estimators to combine

new_preprocess = @(X, Y, W) boost_preprocess_helper(X, Y, W, predict, n_estimators);

end

function [f, Xt] = boost_preprocess_helper( X, Y, W, predict, n_estimators )
%Performs boosting with the given data using SAMME
% link: https://web.stanford.edu/~hastie/Papers/samme.pdf

% characterize X
N = size(X, 1);

% initialize weights with W
boostW = W;

% keep track of model weights
alpha = zeros(1, n_estimators);
% keep track of weighted models
h = cell(1, n_estimators);

% calculate estimators...
for t=1:n_estimators
    % train our model
    [h{t}, Ypred] = predict(X, Y, boostW);
    % compute error
    misclassifications = Ypred ~= Y;
    error = mean(boostW .* misclassifications);
    % compute this model weight
    alpha(t) = log((1-error)/error) + log(4);
    % use the model weight to reweight the samples
    boostW = boostW .* exp(alpha(t) * misclassifications);
    % renormalize W
    boostW = N * boostW / sum(boostW);
end

f = @(inputX) boost_preprocess_combine(inputX, alpha, h);
Xt = f(X);

end

function Xt = boost_preprocess_combine(X, alpha, h)
% predictor for boosting with model weights alpha and models h

Xt = zeros(size(X, 1), 5);
for t=1:length(alpha)
    Xt = Xt + alpha(t) * bsxfun(@eq, h{t}(X), 1:5);
end

end
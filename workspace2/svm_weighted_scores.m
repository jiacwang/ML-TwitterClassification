function [ f, Xt ] = svm_weighted_scores(X, Y, c)
%UNTITLED15 Summary of this function goes here
%   X input matrix
%   Y input labels
%   c cost of violating margin

% function for obtaining cost-aware weights for svm on k-th...
liblinear_params = @(wt_str) sprintf('-s 1 -c %d -B 1 %s -q', c, wt_str);

% we will define implicitly (and expand implicitly :( ) the models obtained
% we will call them model

for k=1:5  % for each possible label
    params = liblinear_params(liblinear_weights(k));  % params for liblinear
    model(k) = liblinear_train(Y, X, params);  % train with the weights
end

% create function for transforming our x into scores...
f = @(inputX) svm_weighted_helper(inputX, model);
% apply it to the training data
Xt = f(X);



end

function wt_str = liblinear_weights(k)
% obtains weights for training on given value of k
costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
w = costs(:, k);
w = w/max(w);
w(k) = 1;
wt_str = sprintf('-w1 %d -w2 %d -w3 %d -w4 %d -w5 %d', w(1), w(2), w(3), w(4), w(5));
end

function Xt = svm_weighted_helper(X, model)
% Xt = repmat(-Inf, [size(X, 1), 5]);  % initialize scores
Xt = zeros(size(X, 1), 5);  % initialize scores
dummy_labels = repmat([1], [size(X, 1), 1]);
for k=1:5  % update Xt with maximum of scores from each model
    [~, ~, scores] = liblinear_predict(dummy_labels, X, model(k), '-b 1');
    % Xt = max(Xt, scores);
    Xt = Xt + scores;
end
end


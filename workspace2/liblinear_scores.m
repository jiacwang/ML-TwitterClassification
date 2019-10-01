function [f, Xt] = liblinear_scores(X, Y, liblinear_options)
%liblinear_scores trains SVM on (X,Y) and returns scores as transformed X
%   X: N x P input matrix
%   Y: N x 1 vector of labels
%   liblinear_options: 'parameters for liblinear_train'
%returns function to transform X to scores and the scores for training

% train model
model = liblinear_train(Y, X, liblinear_options);
% get f
f = @(inputX) liblinear_scores_helper(inputX, model);
% get Xt
Xt = f(X);


end

function Xt = liblinear_scores_helper(X, model)
[~, ~, Xt] = liblinear_predict(repmat([1], [size(X, 1), 1]), X, model, '-b 1');
end


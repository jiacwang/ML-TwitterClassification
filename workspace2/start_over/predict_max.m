function [ f, Yt ] = predict_max( X, ~, ~ )
%PREDICT_MAX Summary of this function goes here
%   X: N x 5 scores for each sample

f = @(inputX) predict_max_helper(inputX);
Yt = f(X);

end

function labels = predict_max_helper(X)
[~, labels] = max(X, [], 2);
end
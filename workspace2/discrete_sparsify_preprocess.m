function [ f, Xt ] = discrete_sparsify_preprocess( X, ~ )
%UNTITLED16 Summary of this function goes here
%   create two columns for every column in X, where we have 1 for greater
%   than 1sd above, -1 for 1sd below, and 0 otherwise

orig_means = mean(X);
orig_stds = std(X);
f = @(inputX) discrete_sparsify_helper(inputX, orig_means, orig_stds);
Xt = f(X);


end

function Xt = discrete_sparsify_helper(X, orig_means, orig_stds)
above = X > orig_means + orig_stds;
below = X < orig_means - orig_stds;
Xt = sparse([X.*above, X.*below]);
end
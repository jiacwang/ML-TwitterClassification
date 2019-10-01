function [ f, Xt ] = preprocess_SVD( X, ~, ~, K )
%PREPROCESS_SVD Computes first K principal components of X without
%weighting
%   X: NxP input matrix
%   K: number of principal components to keep

% get first K components from SVD
[U, S, V] = svds(X, K);
% set PC scores
Xt = U * S;
% in the future, can just project down using V:
f = @(inputX) inputX * V;


end


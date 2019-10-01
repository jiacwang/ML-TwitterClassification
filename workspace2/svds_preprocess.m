function [ f, Xt ] = svds_preprocess( X, ~, K)
%UNTITLED10 Summary of this function goes here
%   X is input matrix
%   ~ is unused input for associated Y vector
%   K is number of components to keep
%Returns f, which calculates transformation for new inputX, Xt is PC scores
%(i.e. U from SVD)

% perform svds on X
[U, S, V] = svds(X, K);
% set PC scores
Xt = U * S;
% we can project down to this space in the future using the other matrices
f = @(inputX) inputX * V;

end


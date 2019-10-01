function [ f, Xt ] = precomputed_SVD( X, ~, k )
%precomputed_SVD uses precomputed SVD on the full dataset
%   Detailed explanation goes here
load('precomputed_SVD.mat');
reducedV = V(:, 1:k);
f = @(inputX) inputX*reducedV;
Xt = f(X);

end


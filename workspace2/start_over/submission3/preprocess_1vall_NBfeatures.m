function [ f, Xt ] = preprocess_1vall_NBfeatures( X, Y, W, k, pseudocount )
%PREPROCESS_1VALL_NBFEATURES Obtains log ratios of NB for k vs rest
%classification
%   X: N x P input matrix
%   Y: N x 1 labels
%   W: N x 1 weights on observations
%   k: scalar {1, 2, 3, 4, 5} which class we are computing these features
%   for
%   pseudocount: pseudocount to smooth probability estimates

% get total observations
N = sum(W);
% get weighted feature counts
WX = W .* (X ~= 0);
% get appearacnes of features for k vs not k
Nj_k = sum(WX(Y==k,:)) + pseudocount;
Nj_notk = sum(WX(Y~=k,:)) + pseudocount;
% get log ratio between these
log_ratio = (log(Nj_k) - log(sum(Nj_k))) - (log(Nj_notk) - log(sum(Nj_notk)));

% transformation is as follows:
f = @(inputX) inputX .* log_ratio;
Xt = f(X);



end


function [ f, Xt ] = preprocess_1vall_NBfeatures( X, Y, W, k, pseudocount, bias )
%PREPROCESS_1VALL_NBFEATURES Obtains log ratios of NB for k vs rest
%classification
%   X: N x P input matrix
%   Y: N x 1 labels
%   W: N x 1 weights on observations
%   k: scalar {1, 2, 3, 4, 5} which class we are computing these features
%   for
%   pseudocount: pseudocount to smooth probability estimates
%   bias: if true, uses log prior on Y as bias vector

if nargin < 6
    bias = false;
end

% get matches of Y to k
matchY = (Y==k);
% get weighted feature counts
WX = W .* (X ~= 0);
% get appearacnes of features for k vs not k
Nj_k = sum(WX(matchY,:)) + pseudocount;
Nj_notk = sum(WX(~matchY,:)) + pseudocount;
% get log ratio between these
log_ratio = (log(Nj_k) - log(sum(Nj_k))) - (log(Nj_notk) - log(sum(Nj_notk)));

% transformation is as follows, adding bias if requested
if bias
    bias_val = log(sum(matchY)) - log(sum(~matchY));
    f = @(inputX) [inputX .* log_ratio, bias_val .* ones(size(inputX, 1), 1)];
else
    % transformation is as follows:
    f = @(inputX) inputX .* log_ratio;
end
% apply it to original X
Xt = f(X);



end


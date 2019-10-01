function [ f, Xt ] = preprocess_shrunken_centroids( X, Y, W, delta )
%PREPROCESS_SHRUNKEN_CENTROIDS Uses Nearest Shrunken Centroids method from
%18.2 in The Elements of Statistical Learning, which is a regularized
%diagonal-shared covariance LDA model (~ NB with diagonal gaussian)
%   X: N x P input matrix
%   Y: N x 1 input vector
%   W: N x 1 weights (assume sum is N)
%   delta: shrinkage parameter

% vectorize Y
Yvec = bsxfun(@eq, Y, 1:5);  % N x 5
% weighted Y
WY = W .* Yvec;  % N x 5
% get number of samples total and per class...
N = sum(W);  % scalar
Nk = sum(WY);  % 1 x 5

% compute centroids
pooled_mean = sum(W .* X) / sum(W); % 1 X P
class_mean = sum((WY./Nk)' * X);  % 5 x P
% compute pooled std
pooled_std = std(X, W);  % 1 x P
% compute prior class probabilities
Pk = Nk / N;  % 1 x 5

% set up our shrinkage...
m = sqrt((1./Nk) - (1/N))';  % 5 x 1, most common classes are less
s0 = median(pooled_std);  % scalar, guard against large d_kj...
% calculate d_kj, the scaled distance a class centroid component away from
% pooled center
d = (class_mean - pooled_mean) ./ (m .* (pooled_std + s0));  % 5 x P

% calculate shrunken d
shrunken_d = sign(d) .* (abs(d) - delta);

% shrunken class centroid (mean)
shrunken_centroid = pooled_mean + m .* (pooled_std + s0) .* shrunken_d;  % 5 x P

% identify relevant features
relevant = max(abs(d)) > 0;
relevant_centroid = shrunken_centroid(:, relevant);
relevant_std = pooled_std(:, relevant);

% compute transformation
f = @(inputX) preprocess_shrunken_centroids_helper(inputX, relevant, relevant_centroid, relevant_std, Pk);
Xt = f(X);


end

function Xt = preprocess_shrunken_centroids_helper(X, relevant, centroids, pooled_std, prior)
% computes the discriminant score for X given relevant centroids (K (=5) x P),
% relevant pooled_std (1 x P), and prior (1 x K). takes X to P relevant
% features

% get only relevant features
relevantX = X(:, relevant);
% get constants
N = size(X, 1);
P = length(relevant);
K = size(centroids, 1);

% compute the difference between x and centroids:
diff = kron(ones(K, 1), relevantX) - kron(centroids, ones(N, 1));
% compute our final discriminant score
Xt = reshape(sum((diff.*diff)/(pooled_std.*pooled_std), 2), N, K) + 2 * log(prior);


end




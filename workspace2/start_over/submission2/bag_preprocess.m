function bagged_preprocess = bag_preprocess( predict, n_estimators, p_samples, p_features, bootstrap )
%BAG_PREDICT Uses predict_ function with bagging procedure to generate
%bagged preprocess -- Nx5 of number of bagged estimators that predict the
%given class
%   predict: @(X, Y, W) --> [f, Yt] predictor
%   n_estimators: how many bagged estimators to combine
%   p_samples: with each bootstrap estimator, how many samples to take?
%   p_features: with each bootstrap estimator, how many features to take?
%   bootstrap: boolean, are we bootstrapping samples or subsampling?

if nargin < 5 || isempty(bootstrap)
    bootstrap = true;
end
if nargin < 4 || isempty(p_features)
    p_features = 1;
end
if nargin < 3 || isempty(p_samples)
    p_samples = 1;
end
if nargin < 2 || isempty(n_estimators)
    n_estimators = 10;
end

bagged_preprocess = @(X, Y, W) bag_preprocess_helper(X, Y, W, predict, n_estimators, p_samples, p_features, bootstrap);

end

function [f, Xt] = bag_preprocess_helper(X, Y, W, predict, n_estimators, p_samples, p_features, bootstrap)
% calculate the number of samples and features
N = size(X, 1);
P = size(X, 2);
draw_sample_ndx = @() randsample(N, max(2,floor(N*p_samples)), bootstrap);
draw_feature_ndx = @() randsample(P, max(2,floor(P*p_features)), false);
% keep track of estimators from each sample
bagged_estimators = cell(1, n_estimators);
% loop over estimators
for ndx=1:n_estimators
    % draw samples and features
    samples = draw_sample_ndx();
    features = draw_feature_ndx();
    % create a prediction function from these samples
    [f, ~] = predict(X(samples, features), Y(samples, :), W(samples, :));
    bagged_estimators{ndx} = @(inputX) f(inputX(:, features));
end
% combine these results
f = @(inputX) bag_preprocess_combine(inputX, bagged_estimators);
Xt = f(X);

end

function Xt = bag_preprocess_combine(X, bagged_estimators)
%BAG_PREPROCESS_COMBINE combines the bagged estimators on the X

Xt = zeros(size(X, 1), 5);
for ndx=1:length(bagged_estimators)
    Xt = Xt + bsxfun(@eq, bagged_estimators{ndx}(X), 1:5);
end

end

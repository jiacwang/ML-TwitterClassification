load('train.mat');
N = size(X_train_bag, 1);
P = size(X_train_bag, 2);

cost = 0.3;
NB_features = true;
weight_labels = false;
preprocess = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), @(X, Y, W) preprocess_logistic(X, Y, W, cost, NB_features, weight_labels));

[f, Xt] = preprocess(X_train_bag, Y_train, ones(size(Y_train)));

%% plot reliability
k = 5;  % which index to plot (1:5)
num_points = 100;  % number of points to consider at a time
normalize = false;
% get started...
% normalize Xt?
if normalize
    Xtn = Xt ./ sum(Xt, 2);
else
    Xtn = Xt;
end
% sort each column separately...
[Xts, ndx] = sort(Xtn, 1);
% sort Y by the selected column
Ys = Y_train(ndx(:, k));
% get average...
splits = floor(N .* (0:num_points) / num_points);
empirical_p = zeros(1, num_points);
for sndx = 1:num_points
    empirical_p(sndx) = mean(Ys((1+splits(sndx)):splits(sndx+1)) == k);
end
% now plot them
plot(linspace(0, 1, N), Xts(:, k));
hold on;
scatter(((1:num_points) - 0.5)./num_points, empirical_p);
hold off;
    
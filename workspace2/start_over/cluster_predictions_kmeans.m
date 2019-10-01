function new_predict = cluster_predictions_kmeans( predict, num_clusters, num_components )
%CLUSTER_PREDICTIONS_KMEANS Creates prediction training function applied to
% num_clusters obtained by k-means on data transformed by SVD with
% num_components components
%   predict: @(X, Y, W) --> [f, Yt], where Yt=f(X)
%   num_clusters: clusters obtained by kmeans
%   num_components: take X to dense representation by SVD/PCA
%Returns @(X,Y,W) --> [f, Yt]...

new_predict = @(X, Y, W) cluster_predict_helper(X, Y, W, predict, num_clusters, num_components);

end

function [f, Yt] = cluster_predict_helper(X, Y, W, predict, num_clusters, num_components)
%CLUSTER_PREDICT_HELPER Applies kmeans to input data and does predictions
%separately on each cluster

% determine our clusters
[f_cluster, idx] = preprocess_clusters(X, num_clusters, num_components);

% train separate models...
f_cell = cell(1, num_clusters);
Yt = zeros(size(Y));
for cluster = 1:num_clusters
    [f_cell{cluster}, Yt(idx==cluster,:)] = predict(X(idx==cluster, :), Y(idx==cluster), W(idx==cluster));
end

% combine the models...
f = @(inputX) cluster_predict_helper2(inputX, f_cluster, f_cell);

end

function Yt = cluster_predict_helper2(X, f_cluster, f_cell)
% f_cluster takes X to cluster indices, which index f_cell, to produce Yt..

% initialize output
Yt = zeros(size(X, 1), 1);
% determine cluster membership
idx = f_cluster(X);
% update Yt per cluster
for cluster = 1:length(f_cell)
    Yt(idx == cluster) = f_cell{cluster}(X(idx==cluster, :));
end
% done
end

function [f, idx] = preprocess_clusters(X, num_clusters, num_components)
% takes X and returns indices of trained clusters (idx) and function to
% take new X and tell me which centroid is closest (f)

% get SVD coordinates
before_cluster = chain_preprocess(@(X, Y, W) preprocess_tfidf(X, [], W, 1), @(X, Y, W) preprocess_SVD(X, Y, W, num_components));
[f_svd, X_svd] = before_cluster(X, [], ones(size(X, 1), 1));

% get clusters (try 10 times
[idx, C] = kmeans(X_svd, num_clusters, 'Replicates', 10, 'MaxIter', 1000);
% get function to tell me which cluster new point belongs to
f = @(inputX) kmeans_predict(f_svd(inputX), C);
end

function idx = kmeans_predict(X, C)
% X is X in format that was originally put into kmeans
% C is output centroids from kmeans
[~, idx] = min(pdist2(X, C), [], 2);
end
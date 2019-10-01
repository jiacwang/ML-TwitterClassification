% workspace.m
% Author: Joseph K Aicher
% testing different models

%% Set up training environment
% load our data
load('train.mat');
N = size(X_train_bag, 1);
P = size(X_train_bag, 2);

% set up cross-validation
K = 10;  % number of folds for evaluation
rng(8213);  % seed for random number generation for evaluation
cv_ndx = crossvalind('Kfold', N, K);

%% Run vanilla Naive Bayes
preprocess = chain_preprocess(@(X, Y, W) preprocess_bernoulliNB(X,Y,W,1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[vanilla_NB_mean, vanilla_NB_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);

%% Try bagging Naive Bayes
preprocess = chain_preprocess(@(X, Y, W) preprocess_bernoulliNB(X,Y,W,1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
bagged = bag_preprocess(predict, 100, 0.8, 1, true);
bagged_predict = chain_predict(@(X, Y, W) predict_max(X, Y, W), bagged);
[bagged_NB_mean, bagged_NB_sd] = evaluate_predict(bagged_predict, X_train_bag, Y_train, cv_ndx, true);

%% Try boosting Naive Bayes
preprocess = chain_preprocess(@(X, Y, W) preprocess_bernoulliNB(X,Y,W,1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
boosted = boost_preprocess(predict, 10);
boosted_predict = chain_predict(@(X, Y, W) predict_max(X, Y, W), boosted);
[boosted_NB_mean, boosted_NB_sd] = evaluate_predict(boosted_predict, X_train_bag, Y_train, cv_ndx, true);

%% Try shrunken centroids method
num_components = 100;
preprocess = chain_preprocess(@(X, Y, W) preprocess_SVD(X, Y, W, num_components), @(X, Y, W) preprocess_shrunken_centroids(X, Y, W, 1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[centroid_mean, centroid_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);

%% Try logistic regression method
num_components = 10;
alpha = 0.1;
lambda = 0.1;
standardize = true;
preprocess = chain_preprocess(@(X, Y, W) preprocess_tfidf(X, Y, W, 1), @(X, Y, W) preprocess_SVD(X, Y, W, num_components), @(X, Y, W) preprocess_elastic(X, Y, W, alpha, lambda, standardize));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[logistic_mean, logistic_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);

%% Try bagging Naive Bayes with stemmed vocabulary
preprocess = chain_preprocess(@(X, Y, W) preprocess_bernoulliNB(X,Y,W,1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
bagged = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), bag_preprocess(predict, 100, 0., 0.8, true));
bagged_predict = chain_predict(@(X, Y, W) predict_max(X, Y, W), bagged);
[stembagged_NB_mean, stembagged_NB_sd] = evaluate_predict(bagged_predict, X_train_bag, Y_train, cv_ndx, true);

%% Try multinomial Naive Bayes
preprocess = chain_preprocess(@(X, Y, W) preprocess_multinomialNB(X,Y,W,1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[multi_NB_mean, multi_NB_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);

%% Bag multinomial Naive Bayes?
preprocess = chain_preprocess(@(X, Y, W) preprocess_multinomialNB(X,Y,W,1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
bagged = bag_preprocess(predict, 100, 0.8, 1, true);
bagged_predict = chain_predict(@(X, Y, W) predict_max(X, Y, W), bagged);
[bagged_NB_mean, bagged_NB_sd] = evaluate_predict(bagged_predict, X_train_bag, Y_train, cv_ndx, true);

%% let's do cross-validation on NB with stemmed vocabulary...
p_samples = [1, 0.8, 0.6];
p_features = [1, 0.8, 0.6];
predictors = [100];
% predictors = [10, 20, 30, 50]; % not a terribly big difference...
% pseudocount = [0.5, 1, 2, 5]; % generally worked best with 1
pseudocount = [1];
for sample = p_samples
    for feature = p_features
        for predictor = predictors
            for pct = pseudocount
                preprocess = chain_preprocess(@(X, Y, W) preprocess_bernoulliNB(X,Y,W,pct));
                predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
                bagged = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), bag_preprocess(predict, predictor, sample, feature, true));
                bagged_predict = chain_predict(@(X, Y, W) predict_max(X, Y, W), bagged);
                [cv_mean, cv_sd] = evaluate_predict(bagged_predict, X_train_bag, Y_train, cv_ndx, false);
                fprintf("pseudocount: %d, bag_size: %d, p_feature: %d, p_sample: %d, error: %f +/- %f\n", pct, predictor, feature, sample, cv_mean, cv_sd);
            end
        end
    end
end

% Output:
% pseudocount: 1, bag_size: 100, p_feature: 1, p_sample: 1, error: 0.948540 +/- 0.026613
% pseudocount: 1, bag_size: 100, p_feature: 8.000000e-01, p_sample: 1, error: 0.940747 +/- 0.025607
% pseudocount: 1, bag_size: 100, p_feature: 6.000000e-01, p_sample: 1, error: 0.947159 +/- 0.033660
% pseudocount: 1, bag_size: 100, p_feature: 1, p_sample: 8.000000e-01, error: 0.943566 +/- 0.024681
% pseudocount: 1, bag_size: 100, p_feature: 8.000000e-01, p_sample: 8.000000e-01, error: 0.939863 +/- 0.029922
% pseudocount: 1, bag_size: 100, p_feature: 6.000000e-01, p_sample: 8.000000e-01, error: 0.946939 +/- 0.033064
% pseudocount: 1, bag_size: 100, p_feature: 1, p_sample: 6.000000e-01, error: 0.946275 +/- 0.030147
% pseudocount: 1, bag_size: 100, p_feature: 8.000000e-01, p_sample: 6.000000e-01, error: 0.947215 +/- 0.032446
% pseudocount: 1, bag_size: 100, p_feature: 6.000000e-01, p_sample: 6.000000e-01, error: 0.952686 +/- 0.032495

%% let's train NB with stemmed vocabulary for submission 2
preprocess = chain_preprocess(@(X, Y, W) preprocess_bernoulliNB(X, Y, W, 1));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
bagged = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), bag_preprocess(predict, 100, 1, 0.8, true));
bagged_predict = chain_predict(@(X, Y, W) predict_max(X, Y, W), bagged);
[f, Yt] = bagged_predict(X_train_bag, Y_train, ones(size(Y_train)));
performance_measure(Yt, Y_train)
save submission2.mat f;

%% try new logistic regression function...
costs = [0.15, 0.2, 0.25, 0.3, .35, 0.4, 0.45, 0.5];
NB_features = true;
weight_labels = false;
for cost = costs
    preprocess = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), @(X, Y, W) preprocess_logistic(X, Y, W, cost, NB_features, weight_labels));
    predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
    [logistic_mean, logistic_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, false);
    fprintf("error: %f +/- %f, cost: %f\n", logistic_mean, logistic_sd, cost);
end

% fairly happy with cost = 0.3 with stemming (0.2 without)...

%% let's train logistic with stemmed vocabulary for submission 3
cost = 0.3;
NB_features = true;
weight_labels = false;
preprocess = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), @(X, Y, W) preprocess_logistic(X, Y, W, cost, NB_features, weight_labels));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[f, Yt] = predict(X_train_bag, Y_train, ones(size(Y_train)));
performance_measure(Yt, Y_train)
save submission3.mat f;

%% let's try logistic with newly implemented bias this time
cost = 0.3;
pseudocounts = [0.1, 0.3, 0.5, 0.7, 1];
for pseudocount = pseudocounts
preprocess = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), @(X, Y, W) preprocess_logisticNB(X, Y, W, cost, pseudocount));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[logisticNB_mean, logisticNB_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);
fprintf("error: %f +/- %f, cost: %f, pseudocount: %f\n", logisticNB_mean, logisticNB_sd, cost, pseudocount);
end

%% compare with original?
cost = 0.3;
NB_features_s = [0.5, 1, 1.2, 1.4];
weight_labels = false;
for NB_features = NB_features_s
preprocess = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), @(X, Y, W) preprocess_logistic(X, Y, W, cost, NB_features, weight_labels));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[logistic_mean, logistic_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);
fprintf("error: %f +/- %f, pseudocount: %f\n", logistic_mean, logistic_sd, NB_features);
end

%% Try logistic with EM on NB weights
cost = 0.3;
pseudocount = 1;
unobs_weight = 0.5;
load('validation.mat');
preprocess = chain_preprocess(@(X, Y, W) preprocess_logisticEM(X, Y, W, cost, pseudocount, X_validation_bag, unobs_weight));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
[logisticEM_mean, logisticEM_sd] = evaluate_predict(predict, X_train_bag, Y_train, cv_ndx, true);

%% Let's try original but split up into clusters?
costs = [0.05, 0.1, 0.3];
pseudocounts = [0.5, 1];
num_clusters_s = [1, 2, 3];
num_components = 100;
for cost = costs
for pseudocount = pseudocounts
for num_clusters = num_clusters_s
preprocess = chain_preprocess(@(X, Y, W) preprocess_stem_vocabulary(X, Y, W), @(X, Y, W) preprocess_logisticNB(X, Y, W, cost, pseudocount));
predict = chain_predict(@(X, Y, W) predict_CPE(X, Y, W), preprocess);
cluster_predict = cluster_predictions_kmeans(predict, num_clusters, num_components);
[cluster_mean, cluster_sd] = evaluate_predict(cluster_predict, X_train_bag, Y_train, cv_ndx, false, 5);
fprintf("error: %f +/- %f, cost: %f, pseudocount: %f, num_clusters: %f\n", cluster_mean, cluster_sd, cost, pseudocount, num_clusters);
end
end
end
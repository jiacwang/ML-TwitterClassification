%% constant parameters
K = 10;  % number of folds
seed = 923;  % seed for random number generator

%% load our data
load('train.mat')
% loads the objects train_raw, X_train_bag, and Y_train
% get bag of words number of observations (N) and covariates (P)
N = size(X_train_bag, 1);
P = size(X_train_bag, 2);

%% obtain indices for K-fold cross-validation
rng(seed);  % initialize random generator here
cv_ndx = crossvalind('Kfold', N, K);  % obtain our seed

CPE_predict = @(Xtrain, Ytrain, Xtest) predict_from_CPE(Xtrain, Ytrain, Xtest);
max_predict = @(Xtrain, Ytrain, Xtest) predict_from_max(Xtrain, Ytrain, Xtest);

%% let's put together logistic regression?
liblinear_logistic = '-s 0 -c 10 -B 1 -q';
logistic_scores = @(X, Y) liblinear_scores(X, Y, liblinear_logistic);
model = chain_model(logistic_scores, @(Xtrain, Ytrain, Xtest) predict_from_CPE(Xtrain, Ytrain, Xtest));
[logistic_mean, logistic_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model);

%% let's put together SVM?
liblinear_SVM = '-s 1 -c 10 -B 1 -q';
SVM_scores = @(X, Y) liblinear_scores(X, Y, liblinear_SVM);
model = chain_model(SVM_scores, @(Xtrain, Ytrain, Xtest) predict_from_max(Xtrain, Ytrain, Xtest));
[SVM_mean, SVM_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% let's do model selection on logistic cost parameter
logistic_cost = [0.1, 0.5, 1, 5];
logistic_mean = logistic_cost;
logistic_std = logistic_mean;
for cost_ndx = 1:length(logistic_cost)
    logistic_str = sprintf('-s 0 -c %d -q', logistic_cost(cost_ndx));
    logistic_scores = @(X, Y) liblinear_scores(X, Y, logistic_str);
    model = chain_model(logistic_scores, CPE_predict);
    [logistic_mean(cost_ndx), logistic_std(cost_ndx)] = evaluate_method(X_train_bag, Y_train, cv_ndx, model);
    fprintf('(C=%d, cost_mean=%f, cost_sd=%f\n', logistic_cost(cost_ndx), logistic_mean(cost_ndx), logistic_std(cost_ndx));
end

%% let's try logistic on transformed data
transform_SVD = @(X, Y) svds_preprocess(X, Y, 50);
%transform_SVD = @(X, Y) precomputed_SVD(X, Y, 200);
%count_words = @(X, Y) number_of_words(X, Y);
%SVD_and_count = union_preprocess(transform_SVD, count_words);
tf_idf = @(X, Y) precomputed_idf(X, Y);
count_words = @(X, Y) number_of_words(X, Y);
tf_idf_counts = union_preprocess(tf_idf, count_words);
SVD_features = chain_preprocess(tf_idf_counts, transform_SVD);
elastic = @(X, Y) lassoglm_scores(X, Y, 0.05, true);
preprocess = chain_preprocess(SVD_features, elastic);
model = chain_model(preprocess, CPE_predict);
[elastic_mean, elastic_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% let's try my attempt at weighted SVM...
count_words = @(X, Y) number_of_words(X, Y);
origin_counts = union_preprocess(count_words);
svm_weighted = @(X, Y) svm_weighted_scores(X, Y, 1);
preprocess = chain_preprocess(origin_counts, svm_weighted);
model = chain_model(preprocess, max_predict);
[wSVM_mean, wSVM_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% let's try SVM on sparsified PCA...
count_words = @(X, Y) number_of_words(X, Y);
origin_counts = union_preprocess(count_words);
transform_SVD = @(X, Y) svds_preprocess(X, Y, 20);
sparsify = @(X, Y) discrete_sparsify_preprocess(X, Y);
svm_weighted = @(X, Y) svm_weighted_scores(X, Y, 1);
preprocess = chain_preprocess(origin_counts, transform_SVD);
preprocess = chain_preprocess(preprocess, sparsify);
preprocess = chain_preprocess(preprocess, svm_weighted);
model = chain_model(preprocess, max_predict);
[wSVM_mean, wSVM_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% let's try random forest with costs
transform_SVD = @(X, Y) svds_preprocess(X, Y, 200);
tf_idf = @(X, Y) precomputed_idf(X, Y);
count_words = @(X, Y) number_of_words(X, Y);
tf_idf_counts = union_preprocess(tf_idf, count_words);
SVD_features = chain_preprocess(tf_idf_counts, transform_SVD);
random_trees = @(Xtrain, Ytrain, Xtest) predict_from_random_forest(Xtrain, Ytrain, Xtest);
model = chain_model(SVD_features, random_trees);
[forest_mean, forest_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% Let's try to stack random forest on top of logistic regression
transform_SVD = @(X, Y) svds_preprocess(X, Y, 50);
tf_idf = @(X, Y) precomputed_idf(X, Y);
count_words = @(X, Y) number_of_words(X, Y);
tf_idf_counts = union_preprocess(tf_idf, count_words);
SVD_features = chain_preprocess(tf_idf_counts, transform_SVD);
elastic = @(X, Y) lassoglm_scores(X, Y, 0.05, true);
preprocess = chain_preprocess(SVD_features, elastic);
% combine with SVD_features without tf-idf done
stacked = chain_preprocess(preprocess, transform_SVD);
random_trees = @(Xtrain, Ytrain, Xtest) predict_from_random_forest(Xtrain, Ytrain, Xtest);
model = chain_model(stacked, random_trees);
[sforest_mean,sforest_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% Let's try to combine different transformations together
tf_idf = @(X, Y) precomputed_idf(X, Y);
count_words = @(X, Y) number_of_words(X, Y);
simple_sentiment = @(X, Y) sentiment_preprocess(X, Y);
stacked_features = union_preprocess(tf_idf, simple_sentiment);
stacked_features = union_preprocess(stacked_features, count_words);
% SVD on these features
transform_SVD = @(X, Y) svds_preprocess(X, Y, 50);
SVD_features = chain_preprocess(stacked_features, transform_SVD);
% elastic net on these features
elastic = @(X, Y) lassoglm_scores(X, Y, 0.1, false);
elastic_features = chain_preprocess(SVD_features, elastic);
% model is using these features with CPE prediction
model = chain_model(elastic_features, CPE_predict);
% evaluate our model with cross-validation
[combined_mean, combined_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);

%% Let's try to do NB
NB = @(X, Y) NB_preprocess(X, Y, 1);
model = chain_model(NB, CPE_predict);
[NB_mean, NB_sd] = evaluate_method(X_train_bag, Y_train, cv_ndx, model, true);
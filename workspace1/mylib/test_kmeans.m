%% load data

load ./data/ocr_train.mat
load ./data/ocr_test.mat

%% pca

[score_train, score_test, numpc] = pca_getpc(X_train, X_test);

% your code to select new features using PCA-ed data
pca_xtrain = score_train(:,1:numpc);
pca_xtest = score_test(:,1:numpc);
size(pca_xtrain)
size(pca_xtest)

% auto encoder

addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
 
% your code to train an Auto-encoder, then learn new features from the original data set
% use rbm.m and newFeature_rbm.m
dbn = rbm(X_train);
[ae_xtrain, ae_xtest] = newFeature_rbm(dbn, X_train, X_test);
size(ae_xtrain)
size(ae_xtest)


% logistic

addpath('./liblinear');
%precision_ori_log = logistic(X_train, Y_train, X_test, Y_test);

% your code to train logistic on PCA-ed and Auto-encoder data
%precision_pca_log = logistic(pca_xtrain, Y_train, pca_xtest, Y_test);
%precision_ae_log = logistic(ae_xtrain, Y_train, ae_xtest, Y_test);

%precision_ori_log 
%precision_pca_log 
%precision_ae_log 

% kmeans

K = [26, 50];
precision_ori_km = zeros(length(K), 1);
for i = 1:length(K)
    k = K(i);
    precision_ori_km(i) = k_means(X_train, Y_train, X_test, Y_test, k);
    
    % your code to train logistic on PCA-ed and Auto-encoder data
	precision_pca_km = k_means(pca_xtrain, Y_train, pca_xtest, Y_test, k);
	precision_ae_km = k_means(ae_xtrain, Y_train, ae_xtest, Y_test, k);

	precision_ori_km 
	precision_pca_km 
	precision_ae_km 
    
end

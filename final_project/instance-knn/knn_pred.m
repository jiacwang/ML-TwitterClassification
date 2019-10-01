function [Y_hat] = knn_pred(X_test_bag, test_raw)

	load knn.mat
	
	[X_test_bag] = pre_processing(X_test_bag, 0);

	n = size(X_test_bag,1);

    costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

	K = 30;

	Y_hat = k_nearest_neighbours(Xtrain2,Ytrain,X_test_bag,K,'l2',costs);

end

function [Y_hat] = predict_labels(X_test_bag, test_raw)


% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
	method = 3;
	if (method == 1)
		Y_hat = k_means_pred(X_test_bag, test_raw);
	elseif (method == 2)
		Y_hat = hmm_pred(X_test_bag, test_raw);
	elseif (method == 3)
		Y_hat = knn_pred(X_test_bag, test_raw);
	end
end

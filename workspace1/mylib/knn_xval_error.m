function [error] = knn_xval_error(X, Y, K, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(X, Y, K, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART), corresponding to the number of folds.
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, K_NEAREST_NEIGHBOURS

% FILL IN YOUR CODE HERE
	n = max(part);
	e(1:n) = 0
	for i = 1:n
		trainmask = part ~= i;
		testmask = ~trainmask;
		xtrain = X(trainmask,:);
		xtest = X(testmask,:);
		ytrain = Y(trainmask,:);
		ytest = Y(testmask,:);
		ytest2 = ( k_nearest_neighbours(xtrain, ytrain, xtest, K, distFunc) );
		sz = size(xtest);
		sz1 = sz(1);
		e(i) = sum(~(ytest2 == ytest));
		e(i) = e(i) / sz1;
	end
	error = sum(e)/n;
end

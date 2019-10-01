function [error] = logistic_xval_error(X, Y, part)
% LOGISTIC_XVAL_ERROR - Logistic regression cross-validation error.
%
% Usage:
%
%   ERROR = logistic_xval_error(X, Y, PART)
%
% Returns the average N-fold cross validation error of the logistic regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, LOGISTIC_REGRESSION

% FILL IN YOUR CODE HERE

	stepsize = 10;
	iterations = 200;

	p = max(part);

	e = zeros(p,1);

	for i = 1:p
		test_mask = part == i;
		train_mask = ~test_mask;
		xtrain = X(train_mask,:);
		xtest = X(test_mask,:);
		ytrain = Y(train_mask,:);
		ytest = Y(test_mask,:);
		yp = logistic_regression(xtrain, ytrain, xtest, stepsize, iterations);
		e(i) = sum(ytest ~= yp)/size(ytest,1);
	end
	error = sum(e)/p;
end

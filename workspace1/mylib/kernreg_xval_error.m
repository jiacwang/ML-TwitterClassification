function [error] = kernreg_xval_error(X, Y, sigma, part, distFunc)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

% FILL IN YOUR CODE HERE
	n = max(part);
	e(1:n) = 0;
	for i = 1:n
		trainmask = part ~= i;
		testmask = ~trainmask;
		xtrain = X(trainmask,:);
		xtest = X(testmask,:);
		ytrain = Y(trainmask,:);
		ytest = Y(testmask,:);
		ytest2 = (kernel_regression(xtrain, ytrain, xtest, sigma));
		sz = size(xtest);
		sz1 = sz(1);
		e(i) = sum(ytest2 ~= ytest)/sz1;
	end
	error = sum(e)/n;
end

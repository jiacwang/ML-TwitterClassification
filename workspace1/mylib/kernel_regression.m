function labels = kernel_regression(Xtrain,Ytrain,Xtest,sigma)

    % Function that implements kernel regression on the given data (binary classification)
    % Usage: labels = kernel_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % sigma : width of the (gaussian) kernel.
    % labels : return an M x 1 vector of predicted labels for testing data.
    

    
    % YOUR CODE GOES HERE

	sz = size(Xtrain);
	n = sz(1);
	p = sz(2);
	sz2 = size(Xtest);
	m = sz2(1);
	for i = 1:m
		s = 0;
		for j = 1:n
			tmp = Xtrain(j,1:p) - Xtest(i,1:p);
			tmp = tmp.^2;
			d2 = sum(tmp);
			k = exp(-d2/(sigma^2));
			s = s + k * (Ytrain(j)*2-1);
		end

		if s >= 0
			labels(i,1) = 1;
		else
			labels(i,1) = 0;
		end
	end
end

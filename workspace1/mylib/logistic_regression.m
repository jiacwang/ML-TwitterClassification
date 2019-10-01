function labels = logistic_regression(Xtrain,Ytrain,Xtest,stepsize,iterations)

    % Function to perform logistic regression on the given data (binary classification)
    % Usage: labels = logistic_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % You may use gradient descent as a subroutine within this function to
    % make things simpler. Use the version of gradient descent
    % (constant step size, variable step size) and the step size which
    % you found to work best empirically.
    
    % Remember: Any modification you might wish to make on the
    % training & testing data must be done here (e.g. adding a new feature).
    % Remember: logistic regression will return probability values, and not
    % the actual labels themselves. This function has to return binary
    % labels, so you will have to perform some thresholding on the computed
    % probability values.
    
    % FILL IN THE REST OF YOUR CODE.
    
	n = size(Xtrain,1);
	p = size(Xtrain,2);
	m = size(Xtest,1);

	Xtrain(:,p+1) = 1;
	Xtest(:,p+1) = 1;
    %[weights,wtf] = gradient_ascent_fixed(Xtrain, Ytrain, stepsize, iterations);
    %[weights2,wtf2] = gradient_ascent_fixed(Xtrain, Ytrain, stepsize, iterations);
    [weights,wtf2] = gradient_ascent_decay(Xtrain, Ytrain, stepsize, iterations);
	
	%{
	xxx = 1:iterations;
	p = plot(xxx,wtf,xxx,wtf2);
	p(1).LineWidth = 2;
	p(2).LineWidth = 2;
	title('error per iteration, improvement, noisy data');
	%}

	for i = 1:m
		xi = Xtest(i,:);
		yp = xi*weights;
		if yp > 0
			labels(i,1) = 1;
		else
			labels(i,1) = 0;
		end
	end

end

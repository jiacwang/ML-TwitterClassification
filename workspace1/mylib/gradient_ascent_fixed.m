function [weights,error_per_iter] = gradient_ascent_fixed(Xtrain,Ytrain,step_size,iterations)
    % Function to perform gradient descent with a fixed step size for logistic regression. 
    % Usage: [weights,error_per_iter] = gradient_descent(Xtrain,Ytrain,step_size,iterations)

    % Xtrain : N x P Matrix of training data, where N is the number of
    % training examples, and P is the dimensionality (number of features)
    
    % Ytrain : N x 1 Vector of training labels (0/1)
    
    % step_size : Step size for gradient descent
    %   You will have to play around with this parameter to ensure good
    %   performance. Too high a step size might cause the gradient descent to
    %   not converge at all, while too small a step size might lead to very slow
    %   convergence. Experiment with this value and set it to the one which
    %   you found to work best empirically. You might find it useful to
    %   plot error_per_iter vs iterations to visualize how your gradient
    %   descent is converging. This would be helpful in determining whether
    %   your step size is too high / too low.
    
    % iterations : Maximum number of iterations
    %   In practice, the number of iterations is not fixed. Instead, we continue
    %   to perform gradient descent until the improvement in training error
    %   or the difference in absolute values of the estimated parameters
    %   (weights) across consecutive iterations drops below a certain
    %   predefined threshold. Sometimes, early stopping is also used as a
    %   form of regularization to prevent overfitting (more on that later).
    %   For the given dataset, your implementation should converge within 500 
    %   iterations. If it doesnt, you are probably doing something wrong, or
    %   have not chosen a good step size.
    
%	threshold = 0.001;
    
	n = size(Xtrain,1);
	p = size(Xtrain,2);

    weights = ones(p,1); % P X 1 vector of initial weights
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!

	step_size = 0.0001;
    
    for iter = 1:iterations
		g =  zeros(p,1);
		en = 0;
		for i = 1:n
			xi = Xtrain(i,:);
			yi = Ytrain(i,1);
			c1 = exp(-(yi*2-1)*(xi*weights));
			c0 = c1*(yi*2-1)/(1+c1);
			g = g + (xi.')*c0;
			yp = xi*weights;
			if yp > 0
				yp = 1;
			else
				yp = 0;
			end
			if (yp ~= yi)
				en = en + 1;
			end
		end
		error_per_iter(iter,1) = en/n;
		s = sqrt(sum(g.^2));
		%g = g/s;
		weights = weights + step_size*g;
    end
end

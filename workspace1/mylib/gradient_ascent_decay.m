function [weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,initial_step_size,iterations)

    % Function to perform gradient descent with a decaying step size for
    % logistic regression.
    % Usage: [weights,error_per_iter] = gradient_descent(Xtrain,Ytrain,step_size,iterations)
    
    % The parameters to this function are exactly the same as the
    % parameters to gradient descent with fixed step size.
    
    % initial_step_size : This parameter refers to the initial value of the step
    % size. The actual step size to update the weights will be a value
    % that is (initial_step_size * some function that decays over time)
    % some good choices for this function might by 1/n or 1/sqrt(n).
    % Experiment with such functions, and initial step size until you get
    % good performance.
    
    % FILL IN THE REST OF THE CODE %
    
	n = size(Xtrain,1);
	p = size(Xtrain,2);

    weights = ones(p,1); % P X 1 vector of initial weights
	weights = weights;
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!

	step_size = initial_step_size;

	ss = zeros(iterations);
	gg = zeros(iterations,p);
    
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
			if yp ~= yi
				en = en + 1;
			end
		end
		error_per_iter(iter,1) = en/n;
		s = sqrt(sum(g.^2));
		g = g/s;
		gg(iter,:) = g;
		weights = weights + step_size*g;
		if iter > 1
			if gg(iter,:)*(gg(iter-1,:).') < 0
				step_size = step_size/2;
			end
		end
    end
end


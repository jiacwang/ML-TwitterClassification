function labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distfunc,costs)

    % Function to implement the K nearest neighbours algorithm on the given
    % dataset.
    % Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % K : number of nearest neighbours used to make predictions on the test
    %     dataset. Remember to take care of corner cases.
    % distfunc: distance function to be used - l1, l2, linf.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % YOUR CODE GOES HERE.
	sz = size(Xtrain);
	n = sz(1);
	p = sz(2);
	sz2 = size(Xtest);
	m = sz2(1);
	k = K;

	count(1:m,5) = 0;
	penalty(1:m,5) = 0;
	labels(1:m,1) = 0;

	if k > n
		return;
	end
	if k < 1
		return;
	end

	if distfunc == 'l1'
		distfunc = @(a,b)sum(a-b);
	elseif distfunc == 'l2'
		distfunc = @(a,b)sqrt(sum((a-b).^2));
	else
		distfunc = @(a,b)max(a-b);
	end
	

	for i = 1:m
		d(1:n,1:2) = 0;
		for j = 1:n
			d(j,1) = distfunc(Xtrain(j,1:p),Xtest(i,1:p));
			d(j,2) = Ytrain(j,1);
			%if d(j,2) == 0
			%	d(j,2) = -1;
			%end
		end
		s = sortrows(d,1);
		mask = s(:,1) <= s(k,1);

		for j = 1:5
			count(i,j) = sum(s(mask,2)==j);
		end

		for j = 1:5
			penalty(i,j) = count(i,:)*costs(:,j:j);
		end
	end

	[~,c] = min(penalty.');
	labels = c.';
	
end


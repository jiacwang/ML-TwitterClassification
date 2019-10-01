function [Y_hat] = k_means_pred(X_test_bag, test_raw)
	load kmeans.mat
	
	[X_test_bag] = pre_processing(X_test_bag, 0);

	n = size(X_test_bag,1);
	K = size(label,1);

	dist = zeros(n, K);

	for i = 1:K
		tmp = bsxfun(@minus, X_test_bag, centroid(i,:));
		tmp = tmp.*100;
		dist(:,i) = sqrt(sum(tmp.^2,2));
	end

    costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
	prob = zeros(n,5);
	penalty = zeros(n,5);

	dist = dist.^-1;
	z = sum(dist,2);
	dist = dist./z;
	labels = repmat(label.',n,1);
	for i = 1:5
		mask = labels == i;
		prob(:,i) = sum(dist.*mask,2);
	end
	for i = 1:5
		penalty(:,i) = prob*costs(:,i);
	end
	dist;
	prob;
	penalty;
	label;
	[~,c] = min(penalty.');
	Y_hat = c.';


	%{
	[~,c] = min(dist.');
	c = c.';
	for i = 1:n
		Y_hat(i,1) = label(c(i));
	end
	size(Y_hat)
	%}
end

function k_means_train(Xtrain, Xvalid, Ytrain)

	Xtrains = pre_processing([Xtrain;Xvalid],1);
	n = size(Xtrain,1);
	p = size(Xtrains,2);
	N = size(Xtrains,1);
	
	ratio = zeros(5,1);
	product = 1;
	for i = 1:5
		ratio(i) = sum(Ytrain == i)/N*5;
		product = ratio(i) * product;
	end
	for i = 1:5
		ratio(i) = product / ratio(i);
	end
	ratio
	%ratio = ratio.^2

	K = 60
	% separate into k clusters and assign labels to each cluster
	label = zeros(K,1);
	centroid = zeros(K,p);
	[IDX,C] = kmeans(Xtrains, K, 'MaxIter', 100);
	for j = 1:K
		mask1 = find(IDX(1:n) == j);
		if (size(mask1,1) == 0)
			label(j) = 5;
		else
			table = tabulate(Ytrain(mask1));
			table2 = table(:,2);
			table2 = table2.*ratio(1:size(table2,1),:);
			[~,index] = max(table2);
			label(j) = table(index,1);
		end

		mask2 = find(IDX == j);
		centroid(j,:) = mean(Xtrains(mask2,:), 1);
	end
	label;
	for i = 1:5
		sum(label == i)
	end

	save kmeans.mat label centroid

end

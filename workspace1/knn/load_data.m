function [Xtrain,Xvalid,Ytrain] = load_data(shrink_ratio)

	load ./train.mat
	load ./validation.mat

	part = make_xval_partition(size(X_train_bag,1), shrink_ratio);
	part2 = make_xval_partition(size(X_validation_bag,1), shrink_ratio);
	selectmask = part == 1;
	selectmask2 = part2 == 1;
	Xtrain = X_train_bag(selectmask,:);
	Ytrain = Y_train(selectmask,:);
	Xvalid = X_validation_bag(selectmask2,:);

	%Xtrain = Xtrain.*10;
	%Xvalid = Xvalid.*10;
end

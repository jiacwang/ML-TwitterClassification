function train_model(Xtrain, Xvalid, Ytrain)
	method = 3;
	if (method == 1)
		k_means_train(Xtrain, Xvalid, Ytrain);
	elseif (method == 2)
		hmm_train(Xtrain, Xvalid, Ytrain);
	elseif (method == 3)
		knn_train(Xtrain, Xvalid, Ytrain);
	end
end


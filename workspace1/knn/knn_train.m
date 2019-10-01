function knn_train(Xtrain, Xvalid, Ytrain)

	Xtrains = pre_processing([Xtrain;Xvalid],1);
	n = size(Xtrain,1);
	p = size(Xtrains,2);
	N = size(Xtrains,1);
	Xtrain2 = Xtrains(1:n,:);
	
	save knn.mat Xtrain2 Ytrain
end

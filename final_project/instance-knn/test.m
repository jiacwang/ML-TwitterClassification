n_folds = 10;
shrink_ratio = 1;
err = zeros(n_folds);

[Xtrain,Xvalid,Ytrain] = load_data(shrink_ratio);
part = make_xval_partition(size(Xtrain,1), n_folds);

%{
for i = 1:5
	sum(Ytrain == i)
end
%}

for i = 1:n_folds
	i
	trainmask = part ~= i;
	testmask = ~trainmask;
	Xtraintrain = Xtrain(trainmask,:);
	Xtraintest = Xtrain(testmask,:);
	Ytraintrain = Ytrain(trainmask,:);
	Ytraintest = Ytrain(testmask,:);
	train_model(Xtraintrain, Xvalid, Ytraintrain);
	Ypred = predict_labels(Xtraintest,[]);
	err(i) = performance_measure(Ypred, Ytraintest)
end

err
mean(err)



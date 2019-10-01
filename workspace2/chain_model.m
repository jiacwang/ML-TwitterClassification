function combined = chain_model( preprocess, model )
%chain_model Chains preprocessing function of X, Y (see chain_preprocess)
%with prediction function @(Xtrain, Ytrain, Xtest) --> labels
%   preprocess: @(X, Y) --> [f, Xt]
%   model: @(Xtrain, Ytrain, Xtest) --> labels
%combined is like model, but implementing preprocess on data first

combined = @(Xtrain, Ytrain, Xtest) chain_helper(Xtrain, Ytrain, Xtest, preprocess, model);


end

function labels = chain_helper(Xtrain, Ytrain, Xtest, preprocess, model)
%chain_helper Chains preprocess and model together...
% train preprocessing function and obtain transformed training data
[f, Xtrain_t] = preprocess(Xtrain, Ytrain);
% apply preprocessing function to transform test data
Xtest_t = f(Xtest);
% apply our model to the transformed features
labels = model(Xtrain_t, Ytrain, Xtest_t);

end

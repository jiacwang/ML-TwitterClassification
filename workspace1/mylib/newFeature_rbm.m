function [new_feat, new_feat_test] = newFeature_rbm(dbn, X_train, X_test)
    % Input: a trained auto-encoder; training dataset; testing dataset 
    % Output: features learned from the input datasets

    % unfold dbn to nn
    nn = dbnunfoldtonn(dbn, size(X_train,2));
    nn.activation_function = 'sigm';

    % Get new training data features
    nn2 = nnff(nn, X_train, X_train);
    new_feat=nn2.a{2};
    
    % Get new test data features
    nn3 = nnff(nn, X_test, X_test);
    new_feat_test = nn3.a{2};

end
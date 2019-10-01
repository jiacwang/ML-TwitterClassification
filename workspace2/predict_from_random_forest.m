function labels = predict_from_random_forest( X_train, Y_train, X_test )
%UNTITLED19 Summary of this function goes here
%   Detailed explanation goes here

costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

% create bag of trees... consider changing size of subsample. consider
% changing maximum depth. Consider chaing to different implementation with
% feature subsets as well.
B = TreeBagger(100, X_train, Y_train, 'Cost', costs);
labels = str2double(predict(B, X_test));

end


function [Y_hat] = predict_labels(X_test_bag, ~)


% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

load('submission3.mat');  % has prediction function f
Y_hat = f(X_test_bag);

end
function [ f, Yt ] = predict_CPE( X, ~, ~ )
%PREDICT_CPE Uses CPE estimates X with cost matrix to predict labels Yt...
%   X: N x 5 CPE (proportional at least) estimates for each sample

% constant cost matrix
costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
f = @(inputX) predict_CPE_helper(inputX, costs);
Yt = f(X);

end

function labels = predict_CPE_helper(X, costs)
expected_costs = X * costs;
[~, labels] = min(expected_costs, [], 2);
end
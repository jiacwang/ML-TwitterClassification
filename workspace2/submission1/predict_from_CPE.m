function labels = predict_from_CPE(~, ~, Xtest)
%predict_from_CPE: produces cost-sensitive hard prediction from cpe_estimate
%   Xtest: N x K(=5) matrix of probabilities (not necessarily
%   normalized)
%returns N x 1 vector of labels

% constant cost matrix
costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];

% compute the expected cost for each sample if we chose to label it each
% index
expected_costs = Xtest * costs;

% get the column index for each row that minimizes expected cost
[~, labels] = min(expected_costs, [], 2);


end


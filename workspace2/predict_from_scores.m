function labels = predict_from_scores(train_scores, train_labels, scores)
%predict_from_scores: trains empirical cost matrix for scores and uses them
%to predict labels
%   train_scores: M x K matrix of scores for training samples
%   train_labels: M X 1 matrix of true labels
%   scores: N x K matrix of scores for test samples
%returns N x 1 vector of labels from scores

objective = @(costs) estimate_objective(costs, train_scores, train_labels);
initial_costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
% optimize costs
costs = fminsearch(objective, initial_costs);
labels = make_prediction(costs, scores);

end

function labels = make_prediction(costs, scores)
[~, labels] = min(scores * costs, [], 2);
end

function objective = estimate_objective(costs, train_scores, train_labels)
%estimate_objective: returns objective to optimize costs with
%add soft constraint to costs using LAMBDA
LAMBDA = 0.1;
unpenalized = performance_measure(make_prediction(costs, train_scores), train_labels);
objective = unpenalized + LAMBDA * sum(sum(costs.*costs));
end


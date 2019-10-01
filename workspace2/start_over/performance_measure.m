function err = performance_measure(y_pred, y_true)
% This function computes the performance measure for the predicted labels
% with respect to the ground truth. The returned error value is a real number
% between 0 and 1.

% y_true: vector of true labels (each label in 1, 2, 3, 4, 5)
% y_pred: vector of predicted labels (each prediction in 1, 2, 3, 4, 5)
% err: cost-sensitive performance

    n = size(y_pred,1);
    if n ~= size(y_true,1)
        'Dimensions of both vectors should be same'
        return;
    end
    
    costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    
    y_pred_mat = bsxfun(@eq, y_pred(:), 1:5);
    y_true_mat = bsxfun(@eq, y_true(:), 1:5);

	err = trace(y_true_mat*costs*y_pred_mat')/n;
end

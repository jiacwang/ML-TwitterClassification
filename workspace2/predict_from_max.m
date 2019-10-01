function labels = predict_from_max( ~, ~, Xtest )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

[~, labels] = max(Xtest, [], 2);

end


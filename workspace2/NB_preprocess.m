function [ f, Xt ] = NB_preprocess( X, Y, pseudocount )
%COST_SENSITIVE_NB performs NB and obtains per observation estimated
%probabilities (without normalization)
%   X input matrix N x P
%   Y labels vector N x 1 (range 1:5)
%   pseudocount add this number of tweets with all the words to each class

N = size(X, 1);  % number of samples (including pseudocounts
Yvec = bsxfun(@eq, Y(:), 1:5);  % vectorize Y
Ny = sum(Yvec);  % 1 x 5 number of occurrences of each...
Njy = ((1*(X>0))' * Yvec) + pseudocount;  % P x 5 number of occurrences of each word in each label

% compute probabilities...
Plabels_log = log(Ny ./ N);
Pwordlabel_log = log(Njy ./ sum(Njy));
% get transform, then apply it
f = @(inputX) NB_preprocess_helper(inputX, Plabels_log, Pwordlabel_log);
Xt = f(X);



end

function Xt = NB_preprocess_helper(X, Plabels_log, Pwordlabel_log)
% helper function for obtaining probabilities for X using previously
% computed log probabilities of labels and conditional log probabilities of
% features being present given label
%   X input matrix to transform
%   Plabels_log: 1 x 5 matrix of log probabilities per class
%   Pwordlabel_log: P x 5 matrix of log probabilities of word given class

% calculate...
sampleP_log = Plabels_log + (1*(X>0)) * Pwordlabel_log;
% now, normalize to largest element
sampleP_log = sampleP_log + max(sampleP_log, 2);
% return this in probability (unnormalized) space
Xt = exp(sampleP_log);


end


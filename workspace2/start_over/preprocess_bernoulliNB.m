function [ f, Xt ] = preprocess_bernoulliNB( X, Y, W, pseudocount )
%PREPROCESS_BERNOULLINB Summary of this function goes here
%   X: N x P matrix of counts
%   Y: N x 1 vector of labels
%   W: N x 1 vector of weights
%   pseudocount: smoothing for unobserved samples

N = sum(W);  % the weighted number of samples
Yvec = bsxfun(@eq, Y(:), 1:5);  % vectorize Y
% Yvec = W .* Yvec;  % weight Yvec appropriately
Ny = sum(W .* Yvec);  % weighted number of samples per label
Njy = ((W.*(X>0))' * Yvec) + pseudocount;  % P x 5 number of occurences of each word in each label

% compute probabilities
Plabels_log = log(Ny ./ N);
Pwordlabel_log = log(Njy ./ sum(Njy));
% get transform, then apply it
f = @(inputX) preprocess_bernoulliNB_helper(inputX, Plabels_log, Pwordlabel_log);
Xt = f(X);

end

function Xt = preprocess_bernoulliNB_helper( X, Plabels_log, Pwordlabel_log )
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



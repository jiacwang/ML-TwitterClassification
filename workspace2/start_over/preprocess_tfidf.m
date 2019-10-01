function [ f, Xt ] = preprocess_tfidf( X, ~, W, pseudocount )
%PREPROCESS_TFIDF Summary of this function goes here
%   Detailed explanation goes here

N = sum(W);
P = size(X, 2);

% compute inverse document frequency (log transformed...
idf = log((N + pseudocount) ./ (sum(W .* (X > 0), 1) + pseudocount / P));  % 1 x P

f = @(inputX) inputX .* idf;
Xt = f(X);


end


function [ f, Xt ] = sentiment_preprocess( X, Y )
%SENTIMENT_PREPROCESS Associates each word with normalized sum of
%associated sentiments
%   Detailed explanation goes here

% vectorize Y
Yvec = bsxfun(@eq, Y(:), 1:5);
% count number of times each word associated with given sentiment
Q = (1*(X > 0))' * Yvec;  % p x 5, p is the vocabulary size
% normalize Q (L2 norm)
Q = Q ./ sqrt(sum(Q.*Q, 2));
% get function to transform X (to nx5)
f = @(inputX) inputX * Q;
% transform original data
Xt = f(X);


end


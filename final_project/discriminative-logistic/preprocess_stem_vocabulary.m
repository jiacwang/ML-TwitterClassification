function [ f, Xt ] = preprocess_stem_vocabulary( X, ~, ~ )
%PREPROCESS_STEM_VOCABULARY Applies precomputed stemming of vocabulary
%   Stemming applied in Pyton using Snowball stemmer

% sparse matrix stem_transform 10000x7366 sparse double
load("stem_vocabulary.mat");
% transformation function
f = @(inputX) inputX * stem_transform;
% apply the transformation
Xt = f(X);


end


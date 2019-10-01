function [f, Xt] = precomputed_idf(X, ~)
%precomputed_idf uses calculated IDF from pooled data to do tf-idf
%computed according to <https://deeplearning4j.org/bagofwords-tf-idf>
load('idf.mat');  % has idf 1 x 10000
f = @(inputX) inputX.*idf;
Xt = f(X);

end
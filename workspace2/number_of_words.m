function [ f, Xt ] = number_of_words( X, ~ )
%number_of_words Calculates the number of words for each observation

f = @(inputX) sum(inputX, 2);
Xt = f(X);


end


function combined = chain_preprocess( step1, step2 )
%chain_preprocess Chains two preprocessing functions of X,Y that return
%transformation function and transformed X and Y together
%   step1: @(X, Y) --> [f, Xt] (Xt = f(X)]
%   step2: @(X, Y) --> [f, Xt] (Xt = f(X)]
% returns combined, which composes these two functions

combined = @(X, Y) chain_helper(X, Y, step1, step2);

end

function [f, Xt] = chain_helper(X, Y, step1, step2)
% helper function that is returned by chain_preprocess with step1 and step2
% masked

[f1, Xt1] = step1(X, Y);
[f2, Xt] = step2(Xt1, Y);
f = @(inputX) f2(f1(inputX));
end


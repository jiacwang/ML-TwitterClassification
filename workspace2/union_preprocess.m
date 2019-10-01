function combined = union_preprocess( step1, step2 )
%union_preprocess Combines two preprocessing functions of X, Y that return
% transformation function and transformed X and Y to extend features
%   step1: @(X, Y) --> [f, Xt] (Xt = f(X)]
%   step2: @(X, Y) --> [f, Xt] (Xt = f(X)]
% if step2 not provided, replaced with identity
%combined is @(X, Y) --> [f, Xt] (f = [step1_f, step2_f], Xt=[Xt_1,Xt_2])
if nargin < 2
    step2 = @(X, Y) identity_preprocess(X, Y);
end

combined = @(X, Y) union_helper(X, Y, step1, step2);

end

function [f, Xt] = union_helper(X, Y, step1, step2)
% perform step1 and step2 on the data independently
[f1, Xt1] = step1(X, Y);
[f2, Xt2] = step2(X, Y);
% combine the features
Xt = [Xt1, Xt2];
f = @(inputX) [f1(inputX), f2(inputX)];

end

function [f, Xt] = identity_preprocess(X, ~)
f = @(X) X;
Xt = X;
end
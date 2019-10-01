function new_predict = chain_predict( predict, preprocess )
%CHAIN_PREDICT Combines predict_ and preprocess steps together
%   predict: @(X, Y, W) --> [f2, Yt], where Yt=f2(X)
%   predict: @(X, Y, W) --> [f1, Xt], where Xt=f1(X)
%Returns @(X,Y,W) --> [f,Yt], where f(X) = f2(f1(X))

new_predict = @(X, Y, W) chain_predict_helper(X, Y, W, predict, preprocess);

end

function [f, Yt] = chain_predict_helper(X, Y, W, predict, preprocess)
%CHAIN_PREDICT_HELPER Applies predict and preprocess to the provided data.
[f1, Xt] = preprocess(X, Y, W);
[f2, Yt] = predict(Xt, Y, W);
f = @(inputX) f2(f1(inputX));
end


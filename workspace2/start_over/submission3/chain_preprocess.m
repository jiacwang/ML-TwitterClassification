function new_preprocess = chain_preprocess( varargin )
%CHAIN_PREPROCESS Takes cell array of preprocessing steps and chains them
%   varargin is a cell array of preprocess functions @(X, Y, W) --> [f,Xt]

new_preprocess = @(X, Y, W) chain_preprocess_list(X, Y, W, varargin{:});


end

function [f, Xt] = chain_preprocess_list( X, Y, W, varargin )
%CHAIN_PREPROCESS_HELPER What chain_preprocess returns...

% base is identity...
f = @(inputX) inputX;
Xt = f(X);
% but if varargin has elements, loop through
if ~isempty(varargin)
    for step_ndx=1:length(varargin)
        [f, Xt] = chain_preprocess_pair(Xt, Y, W, f, varargin{step_ndx});
    end
end
end

function [f, Xt] = chain_preprocess_pair( old_Xt, Y, W, old_f, preprocess )
% adds preprocess to old_Xt and old_f...
[f1, Xt] = preprocess(old_Xt, Y, W);
f = @(inputX) f1(old_f(inputX));

end
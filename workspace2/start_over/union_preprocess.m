function new_preprocess = union_preprocess( keep_original, varargin )
%UNION_PREPROCESS Takes cell array of preprocessing steps and uses them to
%construct combined set of preprocessed steps
%   keep_original is boolean indicating whether to keep original...
%   varargin is cell array of preprocessing steps. Combine resulting features 

new_preprocess = @(X, Y, W) union_preprocess_list(X, Y, W, keep_original, varargin{:});

end

function [f, Xt] = union_preprocess_list( X, Y, W, keep_original, varargin )
%UNION_PREPROCESS_LIST

if isempty(varargin)
    f = @(inputX) inputX;
    Xt = f(X);
else
    [f, Xt] = varargin{1}(X, Y, W);
    for step_ndx=2:length(varargin)
        [f1, Xt1] = varargin{step_ndx}(X, Y, W);
        Xt = [Xt, Xt1];  % would like to preallocate, but final size not known
        f = @(inputX) [f(inputX), f1(inputX)];
    end
    % add on original if requested
    if keep_original
        Xt = [X, Xt];
        f = @(inputX) [inputX, f(inputX)];
    end
end
end


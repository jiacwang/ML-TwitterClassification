function [score_train] = pca_getpc(X_train)

% input: original X for training and testing
% output: PCAed X for training and testing, number of PCs that you selected

cov_train = full(cov(X_train));
[coeff_train, latent, explained] = pcacov(cov_train);
score_train = X_train * coeff_train;
%score_test = X_test * coeff_train;

size(coeff_train)

coeff = coeff_train;

%latent

%index = 1:64

%[cumsum(latent)/sum(latent),index.']

%size(cumsum(latent))

%figure, plot(cumsum(latent)/sum(latent));
%plot(cumsum(explained)/100);
%784 - sum(cumsum(explained) > 85);

% set you numpc here, you should acheive 90% reconstruction accuracy
%numpc = 39;

end

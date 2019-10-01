% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.
%load('../data/X.mat'); % change this to X_noisy if you want to run the code on the noisy data
%load('../data/Y.mat');

N_folds = [2,4,8,16];
errors_xval = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)

its = 20;

KKK = [1,2,3,5,8,13,21,34];
errors_KKK = zeros(its,size(KKK,2));
errors_KKKT = zeros(its,size(KKK,2));

TTT = [1,2,3,4,5,6,7,8,9,10,11,12];
errors_TTT = zeros(its,size(TTT,2));
errors_TTTT = zeros(its,size(TTT,2));

for ki = 1:size(TTT,2)
	for trial = 1:its
		errors = test(X,Y,TTT(ki))
		errors_TTT(trial,ki) = errors(4);
		errors_TTTT(trial,ki) = errors(5);
	end
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_TTT); e = std(errors_TTT); x = TTT; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_TTTT); e = std(errors_TTTT); x = TTT; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, Sigma = [1,2,3,4,5,6,7,8,9,10,11,12]');
xlabel('Sigma');
ylabel('Error');
legend('10-Fold Error','Test Error');
hold off;

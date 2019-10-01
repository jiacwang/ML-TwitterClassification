function dbn = rbm(X_train)

% Input: training dataset
% Output: a trained auto-encoder model

% Train a DBN. Its weights can be used to initialize a NN.
rng(0);

dbn.sizes = [200];
opts.numepochs =   50;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, X_train, opts);
dbn = dbntrain(dbn, X_train, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

end

# CIS 520 Final Project

Authors:

+ Joseph K Aicher
+ Lei Shi
+ Jiachen Wang


Dependencies:

+ [MATLAB]
+ [Python]
+ [NLTK]
+ [liblinear]

   [MATLAB]: https://mathworks.org/products/matlab.html
   [Python]: https://www.python.org
   [NLTK]: http://www.nltk.org/index.html
   [liblinear]: https://www.csie.ntu.edu.tw/~cjlin/liblinear/


## Introduction

For the final project, we have been asked to develop 3 different models to
classify the mood of each tweet in a given set of tweets as either `joy`,
`sadness`, `surprise`, `anger`, or `fear`, such that we minimize a weighted
loss function. These 3 models are of different types:

+ A generative method: Naive Bayes
+ A discriminative method: Logistic Regression
+ An instance-based method: K Nearest Neighbours

We briefly describe the models and how to train them in the following sections.


## Naive Bayes

We implemented Naive Bayes to meet the first baseline (14 hours late,
unfortunately). To be specific, the variant of Naive Bayes we implemented was
multinomial Naive Bayes with counts truncated to 1. We trained it with
a pseudocount of 1 (e.g. Laplace smoothing). We make hard predictions from the
model by picking the label that minimizes the expected cost given the
probability estimate given the input features.

The files to evaluate the model with cross-validation, train our model on the
complete dataset, and predict with the trained model are in `./generative-NB/`.
The script `./generative-NB/script.m` can be run to go through that process.

### Description of files

The following files are in the `./generative-NB/` directory:

+ `chain_predict.m`: function to chain "prediction" step after "preprocessing"
  step to create new combined "prediction" step
+ `chain_preprocess.m`: function to chain list of "preprocessing" steps into
  one step
+ `evaluate_predict.m`: function to evaluate "prediction" step by
  cross-validation
+ `performance_measure.m`: as provided with original project assignment
+ `predict_CPE.m`: "prediction step" on 5 features corresponding to
  un-normalized (or normalized) probability estimates; minimizes expected cost
+ `predict_labels.m`: prediction function using `NB.mat` produced by `script.m`
  in style requested for leaderboard/final submission
+ `preprocess_bernoulliNB`: implements _multinomial_ NB (_not_ Bernoulli,
  despite the name) to output soft probability estimates (e.g. 5 features)
+ `script.m`: as described above, shows cross-validation, training, and
  prediction.
+ `train.mat`: as provided with original project assignment
+ `NB.mat`: saved predictor from `script.m` for use by `predict_labels.m`


## Logistic Regression

We implemented logistic regression with L2 regularization to meet the second
baseline. We train 5 separate binary classification models for each label to
obtain probability estimates which are then naively combined and normalized,
from which we minimize expected cost as previously done for Naive Bayes.
Instead of using the original features, we did the following preprocessing
steps:

+ Stem the vocabulary using `SnowballStemmer` as implemented in the Natural
  Language Toolkit in Python
+ For classification on label k, reweight features according to NB
  log-probability ratios for that feature for the binary classification on that
  label

The files to evaluate the model with cross-validation, train our model on the
complete dataset, and predict with the trained model are in
`./discriminative-logistic/`, and `./discriminative-logistic/script.m`
demonstrates that process. However, because stemming was done in Python, the
following steps must be done first:

+ Run the MATLAB script `./discriminative-logistic/load_vocabulary_hdf5.m` with
  `vocabulary.mat` in the same directory to convert vocabulary file to HDF5
  format (output `./discriminative-logistic/vocabulary.hdf5`
+ Run the Python script `./discriminative-logistic/stem_vocabulary.py` to
  generate sparse linear matrix/transformation `stem_transform` to
  `./discriminative-logistic/stem_vocabulary.mat`

After these steps, we can use `./discriminative-logistic/script.m`.

### Description of files

The following files are in the `./discriminative-logistic/` directory:

+ `chain_predict.m`: as described for Naive Bayes
+ `chain_preprocess.m`: as described for Naive Bayes
+ `evaluate_predict.m`: as described for Naive Bayes
+ `performance_measure.m`: as provided with original project assignment
+ `predict_CPE.m`: as described for Naive Bayes
+ `predict_labels.m`: prediction function using `submission3.mat` produced by
  `script.m` in style requested for leaderboard/final submission
+ `preprocess_1vall_NBfeatures.m`: Transforms input features according to
  log-probability ratios for Naive Bayes applied to binary classification
+ `preprocess_logistic.m`: implements one vs all logistic regression with NB
  weights and L2 regularization using [liblinear] to generate soft probability
  estimates
+ `preprocess_stem_vocabulary.m`: applies transformation produced by
  `stem_vocabulary.py` as saved to `stem_vocabulary.mat` to yield stemmed
  features matrix
+ `script.m`: as described above, shows cross-validation, training, and
  prediction
+ `load_vocabulary_hdf5.m`: script to convert vocabulary file to HDF5 format
  (`vocabulary.hdf5`)
+ `stem_vocabulary.py`: Python script to stem vocabulary found in
  `vocabulary.hdf5` and create feature mapping stored in `stem_vocabulary.mat`
+ `stem_vocabulary.mat`: saved feature mapping produced by `stem_vocabulary.py`
  used by `preprocess_stem_vocabulary.m`
+ `submission3.mat`: saved predictor from `script.m` for use by
  `predict_labels.m`
+ `train.mat`: as provided with original project assignment
+ `vocabulary.hdf5`: original vocabulary in new format, produced by
  `load_vocabulary_hdf5.m` using `vocabulary.mat`
+ `vocabulary.mat`: as provided with original project assignment

## Instance-based method: K Nearest Neighbours

We implemented k nearest neighbours as a reference. We use l2 norm as distance
function. Neighbours' labels are used as indicators of probability a data point
belongs to a class. The class with minimal expected penalty is selected.
By a grid search, we found k = 30 and npc = 20 gives the best performance, which
is hard-coded in the matlab script. Probably due to the nature of this algorithm,
assuming a local relationship in a spheric space, the performance is not very good,
both in time and accuracy. Expect to run for minutes before the result show up.

Since this is an instance based method, there's no training process. This provided
version does a 10-fold cross validation on labeled data. To run the test, simply
run test.m in matlab. You'll see cost of each test and average cost printed.

### Description of files

The following files are in the `./instance-knn/` directory:

+ `train.mat`: labeled training data
+ `validation.mat`: unlabeled training data
+ `k_nearest_neighbours.m`: implementation of knn algorithm
+ `knn_train.m`: read training data, do PCA and save processed data; no real training
+ `knn_pred.m`: a wrapper calling knn algorithm to predict labels according to processed training data
+ `train_model.m`: a selector of training algorithm, in this case knn
+ `predict_labels.m`: a selector of predict algorithm, in this case knn
+ `load_data.m`: literally load data
+ `make_xval_partition.m`: create partition for cross validation
+ `pre_processing.m`: pre-process data, e.g, doing PCA
+ `performance_measure.m`: measure accuracy with cost matrix
+ `knn.mat`: will be generated, processed training data
+ `svd.mat`: will be generated, pc from training data


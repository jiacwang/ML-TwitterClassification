The following is a description of the files:                                                                     

You will be given tweets and your job is to correctly classify them as joy (1), sadness (2), surprise (3), anger (4) and fear (5). We will use a cost-sensitive loss function to calculate the cost of your prediction. You can use the provided performance_measure.m file to calculate the cost of your prediction.


You will get:
Training:
18092 training examples with labels
9098 validation examples without labels


We will have a leaderboard that displays your results and ranks based on the average cost of prediction on the 9098 validation examples without labels.

For your final evaluation we will test your algorithm on a test set of 9310 examples.


File Description:
1) predict_labels.m
	- MATLAB function template giving the format for code submitted to the leaderboard
	  We will call this function on our end to test your algorithms.

2) perfomance_measure.m
	- MATLAB function to calculate the average cost of your prediction based on the cost matrix.


3) vocabulary.mat
	- Contains a 1x10000 vector of strings containing the 10000 words used to generate the bag-of-words features from the raw text. For a tweet, the bag-of-words features are the counts of appearences in the tweet, of words from the vocabulary.

4) train.mat
	- Contains a 18092x1 struct containing the tweets in raw text
	- Contains a 18092x10000 matrix, where the rows correspond to the tweets and columns correspond to the bag-of-words features for the tweets
	- Contains a 18092x1 vector of labels (joy (1), sadness (2), surprise (3), anger (4) or fear (5))

5) validation.mat
	- Contains a 9098x1 struct containing the tweets in raw text
	- Contains a 9098x10000 matrix containing the bag-of-words features for the tweets, where the rows correspond to the tweets and columns correspond to the bag-of-words features for the tweets

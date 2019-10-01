"""
stem_vocabulary.py

Stems given vocabulary...

Author: Joseph K Aicher
"""
# 就是把单词处理一下, https://towardsdatascience.com/pre-processing-in-natural-language-machine-learning-898a84b8bd47
# 比如把后缀ing去掉
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import h5py
from scipy.io import savemat
from scipy.sparse import csr_matrix

# load the vocabulary
vocab = list(h5py.File("vocabulary.hdf5").get("vocabulary"))
# stem the vocabulary
stemmer = SnowballStemmer("english")
vocab_stemmed = [stemmer.stem(str(x)) for x in vocab]
# get map from new vocab to new index...
new_vocab_ndx = dict()
new_locations = [-1 for x in vocab_stemmed]
for ndx, val in enumerate(vocab_stemmed):
    if val not in new_vocab_ndx:
        new_vocab_ndx[val] = len(new_vocab_ndx)
    # add it to locations
    new_locations[ndx] = new_vocab_ndx[val]
# done
# make this a matrix of elements...
row = np.array(list(range(len(new_locations))))
col = np.array(new_locations)
data = np.ones(len(new_locations))
stem_transform = csr_matrix((data, (row, col)), shape=(len(new_locations), len(new_vocab_ndx)))
# save the new sparse matrix
savemat("stem_vocabulary.mat", {"stem_transform": stem_transform})

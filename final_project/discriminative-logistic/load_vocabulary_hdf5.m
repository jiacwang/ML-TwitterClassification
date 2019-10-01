% load the vocabulary file
load('vocabulary.mat');
% convert this into a cell string
vocabulary_cell = cellstr(vocabulary);
% save it to our hdf5 database that can be read by numpy
hdf5write('vocabulary.hdf5', 'vocabulary', vocabulary_cell);
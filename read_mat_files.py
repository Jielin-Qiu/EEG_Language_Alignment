import scipy.io as io

### HOW TO READ MATLAB FILE IN PYTHON 3 ###

# set correct file name
file_name = "/Users/nho/Desktop/resultsZPH_SR.mat"

# index of the array `data` is the number of sentence
data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']

# get all field names for sentence data
print(data[0]._fieldnames)

# example: print sentence
print(data[0].content)

# example: get omission rate of first sentence
omission_rate = data[0].omissionRate
print(omission_rate)

# get word level data
word_data = data[0].word

# get names of all word features
# index of the array `word_data` is the number of the word
print(word_data[0]._fieldnames)

# example: get first word
print(word_data[0].content)

# example: get number of fixations of first word
print(word_data[0].nFixations)

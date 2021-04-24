# Tensorflow throws a warning:- Instructions for updating: Colocations handled automatically by placer.
# We can ignore his warning or shut it, doesnt matter code works fine(for now).

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json, pickle, os, string, random

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.lower()
        word = nltk.word_tokenize(pattern) #Creating a list of words
        words.extend(word)
        docs_x.append(word)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# print(docs_x)
# print()
# print(docs_y)

"""
words: Holds a list of unique words.
labels: Holds a list of all the unique tags in the file.
docs_x: Holds a list of patterns.
docs_y: Holds a list of tags corresponding to the pattern in docs_x.
"""

"""
Type of vectorization we will use: BoW
We’ll be using Bag of Words (BoW) in our code. It basically describes the occurrence of a word within 
a document. In our case, we’ll be representing each sentence with a list of the length of all unique 
words collected in the list “words”. Each position in that list will be a unique word from “words”. If 
a sentence consists of a specific word, their position will be marked as 1 and if a word is not present
 in that position, it’ll be marked as 0.
"""
string_punc = []
for i in string.punctuation:
    string_punc.append(i)
# Stemming
stemmer = LancasterStemmer()
# words = [stemmer.stem(str(w).lower()) for w in words if w not in string_punc]
wording = wordings = []
for item in words:
    if type(item) == 'str':
        wording = [stemmer.stem(w.lower()) for w in words if w not in string_punc]
    elif type(item) == 'list':
        for i in item:
            if i not in string_punc:
                wording.append(i)
        wordings.extend(wording)
    else:
        pass

words = sorted(list(set(wordings)))
labels = sorted(labels)

"""
Similarly, for the output, we’ll create a list which will be the length of the labels/tags we have in 
our JSON file. A “1” in any of those positions indicates the belonging of a pattern in that particular 
label/tag.
"""
training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

# Converting training data into NumPy arrays
training = np.array(training)
output = np.array(output)

# Saving data to disk
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

"""
Now that we’re done with data preprocessing, it’s time to build a model and feed our preprocessed data 
to it. The network architecture is not too complicated. We will be using Fully Connected Layers 
(FC layers) with two of them being hidden layers and one giving out the target probabilities. Hence, 
the last layer will be having a softmax activation.
"""

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch = 200, batch_size = 8, show_metric = True)
model.save("model.tflearn")


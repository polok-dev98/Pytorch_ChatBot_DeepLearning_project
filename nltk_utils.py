#importing the library
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Download the NLTK package 
nltk.download('punkt')

#Initialize the stremmer class
stemmer = PorterStemmer()

#Tokenize the sentance
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# convert into lower case and stemming them
def stem(word):
    return stemmer.stem(word.lower())

#create the Bag of words
def bag_of_words(tokenized_sentence, words):
    
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
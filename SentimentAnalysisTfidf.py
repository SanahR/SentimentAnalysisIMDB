"""
Sentiment Analysis Using the IMDB Dataset

This program is part of my sentiment analysis project, it's one of the many different variations I've made. Although there are 6+ different sentiments, 
the IMDB dataset uses a binary classification system, in which only positive(1), and negative(0) exist. This makes it much easier to work with. 
This program takes a random portion of the IMDB Dataset to train on. It then uses the Tfidf Vectorizer to turn the words into numbers. 

Once the words have been converted into numbers, they are then fed into the KNearestNeighbors Algorithm. The program then searches for other reviews with the same 
words as the review, assigning it a category based on the verdict. After that, the computer's results and the actual results are pitted against each other to find the accuracy score. 

"""
from datasets import load_dataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

########################################
#### CREATING THE DATA #################
########################################

random_index = np.random.randint(0,24999, size = 3000)
imdb_data = load_dataset("stanfordnlp/imdb")
test_corpus_uncut = np.array(imdb_data['test']['text'])
test_corpus_cut = np.array(test_corpus_uncut[random_index])
training_corpus_uncut = np.array(imdb_data['train']['text'])
training_corpus_cut = np.array(training_corpus_uncut[random_index])

########################################
########Creating the Algorithm##########
########################################
vectorizer = TfidfVectorizer()

#########################################
#########Training the Algorithm##########
#########################################
vectorizer.fit(training_corpus_cut)

#########################################
#####Doing the Actual Transformation#####
#########################################
X = vectorizer.transform(training_corpus_cut)
X_train = X.toarray()
X_test = vectorizer.transform(test_corpus_cut)
X_test = X_test.toarray()
#Creating the data for y train
y_train_uncut = np.array(imdb_data['train']['label'])
y_train_cut = y_train_uncut[random_index]
y_test_uncut = np.array(imdb_data['test']['label'])
y_test_cut = y_test_uncut[random_index]

################################################
################################################
####### ACTUAL MACHINE LEARNING PORTION  #######
################################################
################################################

algo = KNN(n_neighbors = 17)
algo.fit(X_train, y_train_cut)
results = algo.predict(X_test)
index = random.randint(0,3000)
percentage = (accuracy_score(y_test_cut,results))*100
#print(results[index])
print("Here's the accuracy score for our algorithm: ", percentage)
print("Your review is ",test_corpus_cut[index])
if int(results[index]) == 1:
  print("We have classified it as positive.")
elif int(results[index]) == 0:
  print("We have classified it as negative. ")

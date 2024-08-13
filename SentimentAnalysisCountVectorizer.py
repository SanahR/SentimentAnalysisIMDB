"""
Sentiment Analysis Using the IMDB Dataset

This is the baseline, original version of the more complicated one using the Tfidf Vectorizer. 
There are two main differences here: 
1. The data isn't random. Instead, it's specifically picked from the middle of the dataset. 
2. Count Vectorizer is used instead of the Tfidf Vectorizer. 

Otherwise, most of the steps were the same. I created the vectorizer and then trained it on the trimmed down version of the training corpus. 
I then did the transformations and converted them into arrays. After that, I created and trained the KNN algorithm. 

To conclude, I compared the computer-predicted results with the actual sentiments using the accuracy score function. I then showed all of these things to the user. 
"""
from datasets import load_dataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

########################################
#### CREATING THE DATA #################
########################################

imdb_data = load_dataset("stanfordnlp/imdb")
test_corpus = imdb_data['test']['text'][11000:14000]
training_corpus = imdb_data['train']['text'][11000:14000]

########################################
########Creating the Algorithm##########
#######################################
vectorizer = CountVectorizer()

#########################################
#########Training the Algorithm##########
#########################################
vectorizer.fit(training_corpus)

#########################################
#####Doing the Actual Transformation#####
#########################################
X = vectorizer.transform(training_corpus)
X_train = X.toarray()
X_test = vectorizer.transform(test_corpus)
X_test = X_test.toarray()
#Creating the data for y train
y_train = imdb_data['train']['label'][11000:14000]
y_test = imdb_data['test']['label'][11000:14000]

################################################
################################################
####### ACTUAL MACHINE LEARNING PORTION  #######
################################################
################################################

algo = KNN(n_neighbors = 17)
algo.fit(X_train, y_train)
results = algo.predict(X_test)
index = random.randint(0,3000)
#print(results[index])
print("Here's the accuracy score for our algorithm: ", accuracy_score(y_test,results))
print("Your review is ",test_corpus[index])
if int(results[index]) == 1:
  print("We have classified it as positive.")
elif int(results[index]) == 0:
  print("We have classified it as negative. ")

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
import os

# Stopwords to remove from the review
path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path = [path]
nltk.download('stopwords', download_dir=nltk.data.path[0])
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Corpus is a collection of text
corpus = []

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    
    # Using set makes it faster than a list (default is list)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    corpus.append(review)

    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# Keep the 1500 most occuring words/features, gives most relevant words for training
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

def add_result(name, cm):
    tp, tn, fp, fn = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
    # Accuracy of the test set (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn)/(tp+tn+fp+fn) # 73% which isn't bad for this dataset size
    # Precision TP / (TP + FP)
    precision = tp / (tp + fp)
    # Recall TP / (TP + FN)
    recall = tp / (tp + fn)
    # F1 Score = 2 * Precision * Recall / (Precision + Recall)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return pd.DataFrame([[name, accuracy, precision, recall, f1_score]], columns=['name', 'accuracy', 'precision', 'recall', 'f1_score'])

results_df = pd.DataFrame()

''' Copy from Naive Bayes lesson '''

# Splitting the dataset into the Training set and Test set
# sklearn.cross_validation is deprecated
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

results_df = results_df.append(add_result('Naive Bayes', cm))


''' Trying other classification models '''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

results_df = results_df.append(add_result('Decision tree', cm))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

results_df = results_df.append(add_result('Random Forest', cm))
# Artificial Neural Network

# Insalling Theano

# Installing Tensorflow

# Installing Keras

# Installed using conda

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # upper bound is excluded in a range
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Since there are 3 categories for country, we need to onehotencode that column
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Refresher:
# OneHotEncoding creates a column for every category value in the specified columns
# The value in each column is either a 1 or 0, meaning the row belongs to that category
# or not

# Avoiding the Dummy Variable Trap by removing one column of the country onehotencoded columns
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
''' 
Tip: # of nodes = avg of (# of nodes in input layer + # of nodes in output layer)
But be an artist, experiment with Parameter tuning, using perhaps k-fold cross validation
Here: (11 + 1) / 2 = 6
Using the rectifier activation function = relu
'''
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
'''
No need for input_dim as the classifier didn't know how many dimensions to expect
in the input layer, now it does.
We'll keep the number of nodes the same still
'''
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
'''
units = output_dim, even in the last two layers, this was the case.
We're using the sigmoid function to get probabilities as output.
If you have a dependent variable that has more than 2 categories, 
like 3 categories: 
1. Change output_dim/units to be the number of classes = 3.
2. The activation function would be 'softmax', which is sigmoid but applied 
to more than 2 categories.
'''
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
'''
Optimizer:
    adam is an implementation of the stochastic gradient descent algo
Loss function:
    It'll be a log loss function because our output layer is sigmoid.
    If your dependent variable has a binary outcome, 
    the logloss function is called 'binary_crossentropy'
    If the dependent variable has more than 2 outcomes, 
    the logloss function is 'categorical_crossentropy'
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
'''
Use the artist within to find a good batch_size and number of epochs
'''
classifier.fit(X_train, y_train, batch_size = 5, epochs = 100)

# Part 3 - making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


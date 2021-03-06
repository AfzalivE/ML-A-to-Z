# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# This function takes as argument your results list and return a tuple list 
# with the format:[(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0])[0] for result in results]
    lh          = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []

for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

# the line creates a date frame which is accessible from Variable explorer
results_df = pd.DataFrame(inspect(results))
results_df.columns = ['0', '1', 'support', 'confidence', 'lift']
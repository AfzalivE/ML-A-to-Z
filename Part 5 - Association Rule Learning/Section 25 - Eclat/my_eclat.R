# Eclat

# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset

# Only consider items bought 4 times a day. Dataset is over 1 week
# So bought 4 * 7 = 21 times a week out of 7500 transactions = 0.0028 ~ 0.003
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Visualizing the results
inspect(sort(rules, by = 'support')[1:10])


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from apyori import apriori

df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

print(df)

print(df.shape)

transactions = []

df_len,df_col = df.shape


for i in range(0,df_len):
    for j in range(0,df_col):
        transactions.append([str(df.values[i,j]) for j in range(0,df_col)])



rules = apriori(transactions=transactions,min_support=0.003, min_confidence = 0.2, min_lift=3, min_length = 2, max_length = 2)

results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]

    return list(zip(lhs,rhs,supports))

resultinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

print(resultinDataFrame)

ara = resultinDataFrame.nlargest(n=10,columns='Support')


def fibonacci_of(n):
    if n in {0, 1}:  # Base case
        return n
    return fibonacci_of(n - 1) + fibonacci_of(n - 2)  # Recursive case

[fibonacci_of(n) for n in range(15)]

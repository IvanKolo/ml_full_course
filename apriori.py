import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from apyori import apriori



df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []

print(df)

print(df.shape)
df_len,df_col = df.shape


for i in range(0,df_len):
    for j in range(0,df_col):
        transactions.append([str(df.values[i,j]) for j in range(0,df_col)])
        # transactions.append([str(df.values[i, j])])



# print(transactions)


rules = apriori(transactions=transactions,min_support=0.003, min_confidence = 0.2, min_lift=3, min_length = 2, max_length = 2)

results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]

    return list(zip(lhs,rhs,supports,confidences,lifts))

resultinDataFrame = pd.DataFrame(inspect(results), columns=['left hand side', 'right', 'support', 'confidence level', 'lift'])

print(resultinDataFrame)

ara = resultinDataFrame.nlargest(n=10,columns='lift')

print(ara)


# alist = [2,4,8,21]
#
# alist[1:4] = [20,14,2]
#
# print(alist)
#
# print(alist[-2])
#
# print(alist[-4:-1])


# import pyautogui
#
# import time
#
# while True:
#     pyautogui.move(0,10)
#
#     time.sleep(2)




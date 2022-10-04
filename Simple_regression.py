import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

# print(X)
# print(Y)

#Spliting data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Training data

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediction the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

#Visualising the Training set results

plt.scatter(X_train, Y_train, c = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Expirience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, Y_test, c = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Expirience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


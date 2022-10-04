import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:,-1].values
# print(X)
# print(Y)

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
print(X_poly)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X,Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth in the ass of Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

### Polynomial regression
# plt.scatter(X,Y, color = 'red')
# plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
# plt.title('Truth in the ass of Polynomial Regression')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

### Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color = 'blue')
plt.title('Truth in the ass of Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


predicted_salary = lin_reg.predict([[6.5]])
print(predicted_salary)

predicted_salary = lin_reg_2.predict(poly_reg.transform([[6.5]]))
print(predicted_salary)
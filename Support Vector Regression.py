import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)

y = y.reshape(len(y), 1)
print(y)

sc_x = StandardScaler()
sc_y = StandardScaler()


### Data Preproccessing
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)
print(X)
print(Y)

regressor = SVR(kernel= 'rbf')
regressor.fit(X,Y)

predicted_salary = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print('predicted_salary', predicted_salary)

### Visualising the SVR results
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y), color = 'red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth in the ass of SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

### Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid))), color = 'blue')
plt.title('Truth in the ass of SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
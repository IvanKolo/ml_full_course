import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv('Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

print(len(x))
print(y)

regressor = RandomForestRegressor(n_estimators=10, random_state= 0)
regressor.fit(x, y)

prediction = regressor.predict([[6.5]])
print(prediction)

X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth in the ass of Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



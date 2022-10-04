from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC



dataset = pd.read_csv('Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
# print(X)
# print(Y)

sc = StandardScaler()



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print('X_train_before', X_train)
print('X_test_before', X_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('X_train_after', X_train)
print('X_test_after', X_test)

classifier = DecisionTreeClassifier(criterion= 'entropy', random_state= 0)
classifier.fit(X_train,Y_train)

print(sc.transform([[30, 87000]]))

predict = classifier.predict(sc.transform([[30, 87000]]))
print(predict)

Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))


matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
print(matrix)
print('Accuracy', accuracy)

### Visualising the Training Set

X_set, Y_set = sc.inverse_transform(X_train), Y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 10, stop = X_set[:,0].max() + 10, step = 0.25),
                    np.arange(start = X_set[:,1].min() - 1000, stop = X_set[:,1].max() + 1000, step = 0.25))
plt.contourf(X1,X2, classifier.predict(sc.transform(np.array([X1.ravel(),X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('red', 'green'))(i),label = j)
plt.title('Decision Tree Estimation (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

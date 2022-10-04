import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv('Mall_Customers.csv')

print(data.info)
print(data.describe())

X_ = data[['Annual Income (k$)','Spending Score (1-100)']]

X = data.iloc[:,[3,4]].values

# X = data[['Annual Income (k$)','Spending Score (1-100)']].values

# print(X['Annual Income (k$)'])
#
plt.scatter(X_['Annual Income (k$)'], X_['Spending Score (1-100)'])
plt.show()

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidaen Distance')
plt.show()


hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,0],X[y_hc == 0, 1], s = 100, c = 'red', label= 'Cluster1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1, 1], s = 100, c = 'blue', label= 'Cluster2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2, 1], s = 100, c = 'green', label= 'Cluster3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3, 1], s = 100, c = 'cyan', label= 'Cluster4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4, 1], s = 100, c = 'magenta', label= 'Cluster5')
# plt.scatter(X[y_hc == 5,0],X[y_hc == 5, 1], s = 100, c = 'yellow', label= 'Cluster5')
plt.title('Cluster of Customers')
plt.xlabel('Annaual Income k$')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# n_samples, n_features = X.shape
#
# print(n_features, n_samples)
#
# weightns, bias = np.zeros(n_features), 0
#
# print(weightns, bias)

def myf(default = []):
    default.append('python')
    return default


print(myf())

print(myf())


df = pd.DataFrame(np.random.randn(5,3))
df.style.set_table_styles([{'selector': 'tr:hover', 'props': [('bacground_color','yellow')]}])

print(df)


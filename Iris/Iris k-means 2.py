import numpy as np
import pandas as pd
import scipy
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Creating dataFrame
d1 = datasets.load_iris()
df = pd.DataFrame(data=np.c_[d1['data'], d1['target']],
                  columns=d1['feature_names'] + ['target']).reset_index()
df1 = df.groupby(['target'])  # not really used somewhere

pd.set_option('display.expand_frame_repr', False)

# Groupby Class and characteristics
dfGrSL = df.groupby(['target']).mean()
dfGrSD = df.groupby(['target']).std()  # sepal length, and petal length + width show significant differences in sd
# Plot
dfPL1 = df.plot(kind='scatter', x='petal length (cm)', y='petal width (cm)', c='target', colormap='cool')
dfPL2 = df.plot(kind='scatter', x='sepal length (cm)', y='petal length (cm)', c='target', colormap='cool')
dfPL3 = df.plot(kind='scatter', x='sepal length (cm)', y='petal width (cm)', c='target', colormap='cool')

df['Ratio1'] = df['sepal length (cm)']/df['sepal width (cm)']
df['Ratio2'] = df['petal length (cm)']/df['petal width (cm)']
dfPL4 = df.plot(kind='scatter', x='Ratio1', y='Ratio2', c='target', colormap='cool')

# Modeling/Clustering

X = np.matrix([df['sepal length (cm)'].values,
     df['sepal width (cm)'].values,
     df['petal length (cm)'].values,
     df['petal width (cm)'].values])

X1 = X.T

kmeans = KMeans(n_clusters=3, init='k-means++').fit(X1)

dfPL5 = df.plot(kind='scatter', x=X1[:, 0], y=X1[:, 1], c=kmeans.labels_.astype(float), colormap='cool')

plt.figure()

plt.scatter(X1[:,1], X1[:,0], c=kmeans.labels_.astype(float))















import numpy as np
import pandas as pd
import scipy
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Creating dataFrame
d1 = datasets.load_iris()
df = pd.DataFrame(data = np.c_[d1['data'], d1['target']],
                  columns = d1['feature_names'] + ['target']).reset_index()
df1 = df.groupby(['target']) # not really used somewhere

pd.set_option('display.expand_frame_repr', False)


def kMeanz(dataframez, column_list):

    # Creating column list
    intz = []
    for i in column_list:
            va = int(i)
            intz.append(va)

    X = dataframez.iloc[:, intz].values
    # How  to transpose: X1 = X.T

    # Shoulder Method (finding optimum number of clusters), Plotting results, Automatic detection of clusters
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbowing')
    plt.xlabel('No. Clusters')
    plt.ylabel('WCSS') # euclidian distance
    XY = plt.gca().get_lines()[0].get_xydata()
    plt.close()

# Distance method for clusters
    XYLength = 10
    XYFirst = XY[0]
    XYLineVec = XY[-1] - XY[0]
    XYVecNorm = XYLineVec / np.sqrt(np.sum(XYLineVec ** 2))
    XYVecFromFirst = XY - XYFirst
    XYScalarProduct = np.sum(XYVecFromFirst * np.matlib.repmat(XYVecNorm, XYLength, 1), axis=1)
    XYVecFirstParallel = np.outer(XYScalarProduct, XYVecNorm)
    XYVecToLine = XYVecFromFirst - XYVecFirstParallel
    XYDistToLine = np.sqrt(np.sum(XYVecToLine ** 2, axis=1))
    XYIdxOfBestPoint = np.argmax(XYDistToLine)
    XYNo = XYIdxOfBestPoint + 1


    kmeans = KMeans(n_clusters=XYNo, init='k-means++')
    y_kmeans = KMeans(n_clusters=XYNo, init='k-means++').fit(X)
    x_kmeans = y_kmeans.predict(X)

    clusterz = plt.scatter(X[:,0], X[:,1], c=x_kmeans, s=50, cmap='cool')

    return clusterz






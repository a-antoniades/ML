unique_infoimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
import statistics
import re
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.spatial.distance import mahalanobis
import scipy as sp
import numpy as np


footprint = pd.read_csv("C:/PythonFolder/to_test_footprint1.csv")
mahalanobis_dataset = pd.read_csv("C:/PythonFolder/mahalanobis_1.csv")
final_metrics = pd.read_csv("C:/PythonFolder/final_metrics.csv")

#Define the MAD function
def MAD(data, coeff, plot=False):
    med = np.median(data)
    #Thelei for loop to MAD gia oles tis
    mad1 = np.median(abs(data-med))
    upper_lim = med + coeff*mad1
    lower_lim = med - coeff*mad1
    if lower_lim < 0:
        lower_lim = 0          #If lower limit is smaller than 0, replace it with 0
    #Anomaly Detection Step
    result = []
    for i in data:
        if i > upper_lim:
            result.append(1)
        else:
            result.append(0)
    #Plot function
    if(plot):
        xRange = []
        plt.figure()
        for i in range(1, len(data)+1, 1):
            xRange.append(i)
        plt.fill_between(xRange, upper_lim, lower_lim, color='0.9')
        plt.scatter(xRange,data, marker = 'o', c = result)
        plt.plot(xRange,data, linestyle = '--', color = 'blue')
        plt.ylabel('number of ...')
        plt.show()
    else:
        return result

def anomalyIQR(data, coeff, plot=False):
    #calculate the median of the dataset
    med = np.median(data)
    #calculate the 25 and 75% percentile quartiles
    q75, q25 = np.percentile(data,[75,25])
    #Calculate the upper limit for the outlier detection ranges
    upperLim = med + coeff*(q75-q25)
    lowerLim = med - coeff*(q75-q25)
    if lowerLim < 0:
        lowerLim = 0 #If lower limit is smaller than 0, replace it with 0
    #Declare empty result list and insert 1 if above limit, 0 otherwise
    result = []
    for i in data:
        if i > upperLim:
            result.append(1)
        else:
            result.append(0)
    #Plot or return result
    if(plot):
        xRange = []
        plt.figure()
        for i in range(1, len(data) + 1, 1):
            xRange.append(i)
        plt.fill_between(xRange, upperLim, lowerLim, color='0.9')
        plt.scatter(xRange, data, marker='o', c=result)
        plt.plot(xRange, data, linestyle='--', color='blue')
        plt.ylabel('number of ...')
        plt.show()
    else:
        return pd.DataFrame({'Score':data, 'Thr.': upperLim ,'anomaly': result})

def mahalanobis1(X):
    #Takes a panda df as input with ONLY the numeric variables that will calculate the distance
    covX = X.cov().values
    covX = sp.linalg.inv(covX)
    mean1 = X.mean().values
    return (np.sqrt(np.sum(np.dot((X - mean1), covX) * (X - mean1), axis=1))**2)

def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return(result) 

def scatter_fun(data, limits, plot=False):
    counts = data.value_counts()
    min_cnt = min(counts)
    score = (min_cnt/counts)*100

    df = pd.concat([counts, score], axis = 1).reset_index()
    cols = list(df.columns)
    cols[-2] = 'Frequency'
    cols[-1] = 'Score'
    cols[-3] = 'Variable'
    df.columns = cols

    r = df['Score']
    theta = df.index.values

    #Thresholds
    Thr1 = anomalyIQR(df['Score'], limits[0])['Thr.'][0]
    Thr2 = anomalyIQR(df['Score'], limits[1])['Thr.'][0]
    Thr3 = anomalyIQR(df['Score'], limits[2])['Thr.'][0]

    #Figure
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = 'polar')
        #add plots
        ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*Thr1, 
                c='r', ls='--', alpha = 0.5)
        ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*Thr2, 
                c='y', ls='--', alpha = 0.5)
        ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*Thr3, 
                c='g', ls='--', alpha = 0.5)

        #Plot
        c = ax.scatter(theta, r,
                color=['g' if val < Thr3 else 'y' 
                       if val < Thr2 else 'orange' 
                       if val <Thr1 else 'r' for val in r],
                alpha = 0.7)
        plt.show()
    else:
        result = pd.DataFrame({
            'data': df['Variable'],
            'Score':df['Score'],
            'Relaxed':anomalyIQR(df['Score'], limits[0])['anomaly'],
            'Relaxed Threshold' : Thr1,
            'Normal': anomalyIQR(df['Score'], limits[1])['anomaly'],
            'Normal Threshold' : Thr2,
            'Aggressive': anomalyIQR(df['Score'], limits[2])['anomaly'],
            'Aggressive Threshold' : Thr3
            })
        result = result[['data', 'Score', 'Relaxed', 'Relaxed Threshold',
                         'Normal', 'Normal Threshold',
                         'Aggressive', 'Aggressive Threshold']]
        return(result)

def predict(x, span, periods):
    x_predict = np.zeros((span+periods,))
    x_predict[:span] = x[-span:]
    pred = pd.stats.moments.ewma(x_predict, span)[span:]
    return pred

def load_csvs(path):
    files = glob.glob(path + "/*.csv")
    for counter, file in enumerate(files):
        if counter == 0:
            results = pd.read_csv(file)
        else:
            df_temp = pd.read_csv(file)
            results = results.append(df_temp)
            del(df_temp)
        if counter%10 == 0:
            perc = floor(100*(counter/len(files)))
            print( str(perc) + '% [' + floor(perc/2)*'=' + (50-floor(perc/2))*' ' + ']' )
    return results

import numpy as np
%matplotlib inline
from sklearn import datasets, cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# calculate distance between two d-dimensional points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum([(c1 - c2)**2 for c1, c2 in zip(p1, p2)]))

def find_closest_centroid(datapoint, centroids):
    # find the index of the closest centroid of the given data point.
    return min(enumerate(centroids), key=lambda x: euclidean_distance(datapoint, x[1]))[0]

def randomize_centroids(data, k):
    random_indices = np.arange(len(data))
    np.random.shuffle(random_indices)
    random_indices = random_indices[:k]
    centroids = [data[i] for i in range(len(data)) if i in random_indices]
    return centroids

MAX_ITERATIONS = 10

# return True if clusters have converged , otherwise, return False  
def check_converge(centroids, old_centroids, num_iterations, threshold=0):
    # if it reaches an iteration budget
    if num_iterations > MAX_ITERATIONS:
        return True
    # check if the centroids don't move (or very slightly)
    distances = np.array([euclidean_distance(c, o) for c, o in zip(centroids, old_centroids)])
    if (distances <= threshold).all():
        return True
    return False

def update_centroids(centroids, clusters):
    assert(len(centroids) == len(clusters))
    clusters = np.array(clusters)
    for i, cluster in enumerate(clusters):
        centroids[i] = sum(cluster)/len(cluster)
    return centroids

def kmeans(data, k=2, centroids=None):
    
    data = np.array(data)
    # randomize the centroids if they are not given
    if not centroids:
        centroids = randomize_centroids(data, k)

    old_centroids = centroids[:]

    iterations = 0
    while True:
        iterations += 1

        # init empty clusters
        clusters = [[] for i in range(k)]

        # assign each data point to the closest centroid
        for datapoint in data:
            # find the closest center of each data point
            centroid_idx = find_closest_centroid(datapoint, centroids)
            
            # assign datapoint to the closest cluster
            clusters[centroid_idx].append(datapoint)
        
        # keep the current position of centroids before changing them
        old_centroids = centroids[:]
        
        # update centroids
        centroids = update_centroids(centroids, clusters)
        
        # if the stop criteria are met, stop the algorithm
        if check_converge(centroids, old_centroids, iterations):
            break
    
    return centroids

import compileall
compileall.compile_dir('E:\\PythonFolder\\Compile_Normalize_Score')

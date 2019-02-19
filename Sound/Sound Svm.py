import wave
import struct
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost import XGBClassifier


# Viewing Full DF
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# Converting sound into numbers
def pcm_channels(wave_file):
    """Given a file-like object or file path representing a wave file,
    decompose it into its constituent PCM data streams.

    Input: A file like object or file path
    Output: A list of lists of integers representing the PCM coded data stream channels
        and the sample rate of the channels (mixed rate channels not supported)
    """
    stream = wave.open(wave_file, "rb")

    num_channels = stream.getnchannels()
    sample_rate = stream.getframerate()
    sample_width = stream.getsampwidth()
    num_frames = stream.getnframes()

    raw_data = stream.readframes( num_frames ) # Returns byte data
    stream.close()

    total_samples = num_frames * num_channels

    if sample_width == 1:
        fmt = "%iB" % total_samples # read unsigned chars
    elif sample_width == 2:
        fmt = "%ih" % total_samples # read signed 2 byte shorts
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    integer_data = struct.unpack(fmt, raw_data)
    del raw_data # Keep memory tidy (who knows how big it might be)

    channels = [[] for time in range(num_channels)]

    for index, value in enumerate(integer_data):
        bucket = index % num_channels
        channels[bucket].append(value)

    return channels


# Joining all recordings
def sound_array(folder):
    soundArray = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            sound = pcm_channels(os.path.join(folder, filename))
            soundArray.append(sound[0])

    return soundArray


# Making all rows equal
def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)

    return out


# Creating dataframe
def make_df(directory):
    soundArray = sound_array(directory)
    soundArray = np.array(soundArray)
    soundArray = numpy_fillna(soundArray)
    soundArraydf = pd.DataFrame(soundArray)

    return soundArraydf


soundArray0 = make_df("/Users/antonis/Desktop/Themas/Sound/sound samples")

#soundArray0['Y'] = None
#soundArray0['Y'].iloc[0:50, ] = 0
#soundArray0['Y'].iloc[50:101, ] = 1

# dfXtrain1, dfYtrain1, dfXtest1, dfYtest1
dfX1random = soundArray0.iloc[0:50, 0: 225281]
dfX1random = dfX1random.sample(frac=1, random_state=69).reset_index(drop=True)
dfX1full = dfX1random.append(soundArray0.iloc[50:101, ])
dfX1X = dfX1full.iloc[:, 0:25281]
yLabel = dfX1full['Y']

outDF = pd.DataFrame(index=range(int(25280/3)))  # BUG HERE NEED VARIABLE FOR SIZE
outDF = pd.DataFrame(outDF)
outDF = outDF.append(tripleCol, axis=1, ignore_index=False)
dataframe = dataframe.add(18986)
dataframe = dataframe.multiply(2)
outDF = np.zeros(shape=(101, 1))

outDF = np.hstack((outDF, tripleCol))
outDF = pd.DataFrame(outDF)
outDF = np.hstack((outDF, tripleCol))
outDF = outDF.append(tripleCol, ignore_index=True)
tripleCol = np.hstacktripleCol.sum(axis=0)

outDF = pd.DataFrame(outDF)
outDF = outDF.add(18986)
outDF = outDF.multiply(2)
# Normalize
i = 0
def normalize(dataframe):
    outDF = np.empty((101,0))
    dataframe = dataframe.values
    dataframe = np.array(dataframe)
    for i in range(dataframe.shape[1]):
        tripleCol = np.sum((dataframe[:, 0], dataframe[:, 1], dataframe[:, 2]), axis=0)
        tripleCol = np.expand_dims(tripleCol, axis=1)
        outDF = np.hstack((outDF, tripleCol))
        i += 3


    return outDF
DF = normalize(soundArray0)




dfXtrain1 = dfX1full.iloc[0:36, 0:225280]
dfYtrain1 = dfX1full.iloc[0:36, 225280]

dfXtest1 = dfX1full.iloc[36:66, 0:225280]
dfXtest2 = dfX1full.iloc[36:51, 0:225280]

# Diluted Set

dfX1full1 = dfX1full[dfX1full.columns[::3]]
dfX1full1.columns = range(dfX1full1.shape[1])
dfXtrain11 = dfX1full1.iloc[0:36, 0:75094]
dfXtest11 = dfX1full1.iloc[36:66, 0:75094]

dfMax = soundArray0.max(axis=1)
dfMax = dfMax.max(axis=0)

dfMin = soundArray0.min(axis=1)
dfMin = dfMin.min(axis=0)

dfCheck = soundArray0.iloc[3]


# One-Class SVM
clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma='scale')
clf.fit(dfXtrain11)
pred = clf.fit_predict(dfXtest11)
y_pred = clf.predict(dfXtest11)

# Order by column curve, orderby, rank
import pandas as pd
from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from graphviz import Source
from subprocess import check_call
import matplotlib.pyplot as plt
import pydot

# Storing dataframe

dftrainz = pd.read_csv("/Users/antonis/Desktop/OdysseyProjects/OD4 Titanic/train.csv")
dftestz = pd.read_csv("/Users/antonis/Desktop/OdysseyProjects/OD4 Titanic/test.csv")
dfSurvive = pd.read_csv("/Users/antonis/Desktop/OdysseyProjects/OD4 Titanic/titanic3.csv")
pd.set_option('display.expand_frame_repr', False)

df = pd.concat([dftrainz, dftestz])
df = df.reset_index()

# If you want to print whole df: print(df.to_string())

# Tidying up

df['Sex'] = df['Sex'].str.replace('male', '1')
df['Sex'] = df['Sex'].str.replace('fe1', '0')

df['Sex'] = pd.to_numeric(df['Sex'])  # Convert str to int

# Replace NaN with average value
dfAgeMean = df['Age'].mean()
df['Age'] = df['Age'].fillna(dfAgeMean)

# Split Ticket column

df = df[df.Ticket.str.contains("LINE") == False]

def separate(keyword):

    if ' ' in keyword:
        Ticket11 = keyword.split(" ")[-2]
        Ticket12 = keyword.split(" ")[-1]
        return Ticket11, Ticket12
    else:
        Ticket11 = None
        Ticket12 = keyword
        return Ticket11, Ticket12


dfPantelis = df['Ticket'].apply(separate)
dfPantelis1 = dfPantelis.apply(list)

df['Ticketz'] = dfPantelis1
df[['Ticket1', 'Ticket2']] = pd.DataFrame(df['Ticketz'].values.tolist(),  # column split list into two columns
                                          index=df.index)

df['Ticket2'] = pd.to_numeric(df['Ticket2'])

# Tidying and one-hot encoding 'Embarked'

df['Embarked'] = df['Embarked'].str.replace('S', '1')
df['Embarked'] = df['Embarked'].str.replace('C', '2')
df['Embarked'] = df['Embarked'].str.replace('Q', '3')

df[['Embarked_S', 'Embarked_C', 'Embarked_Q']] = pd.get_dummies(df['Embarked'])

cols = df.columns.tolist()
df = df[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
        'Embarked', 'Embarked_S', 'Embarked_C', 'Embarked_Q', 'Ticketz', 'Ticket1', 'Ticket2']]

# Merging test data survive columns

dfSurvive = dfSurvive.drop([1309])
dfSurviveCols = dfSurvive.columns.tolist()
dfSurvive = dfSurvive.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
                           'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'], axis=1)

df = pd.merge(df, dfSurvive, on='Name')
dfCols = df.columns.tolist()
dfCols = ['PassengerId', 'Survived', 'Survived2', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
          'Cabin', 'Embarked', 'Embarked_S', 'Embarked_C', 'Embarked_Q', 'Ticketz', 'Ticket1', 'Ticket2']
df = df[dfCols]

# Replacing values in 'Fare' and converting to numeric

df['Fare'] = df['Fare'].fillna(value='15')
df['Fare'] = df['Fare'].apply(pd.to_numeric)

# Plotting relationships

dfP = df[dfCols]
graphTicks = df.plot(kind='scatter', x='Ticket2', y='Survived2', c='Survived2', colormap='cool').set_xlim(0, 500000)
PClassCount = df.groupby(['Pclass', 'Survived2'], as_index=False).count()
graphPclass = PClassCount.plot(kind='bar', x='Pclass', y='PassengerId', colormap='cool')

meanFares = df.groupby(['Pclass'], as_index=False).mean()
graphFares = df.plot(kind='scatter', x='Fare', y='Pclass', c='Pclass', colormap='inferno').set_xlim(0, 280)

# Splitting data frame into training/testing

dftrain = df.iloc[0:891]
dftest = df.iloc[893:]


# -- Running Models

# Splitting train/test

colz = dftrain.columns.tolist()
colz = ['PassengerId', 'Survived', 'Survived2', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Ticketz',
        'Embarked_S', 'Embarked_C', 'Embarked_Q', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket2', 'Ticket1']

# Choose which columns to use

dfXtrain = dftrain[colz].iloc[:, 8:18]
dfYtrain = dftrain[colz].iloc[:, 2]


def numberfy(dataframe):
    for x in dataframe.columns:
        dataframe[x].apply(pd.to_numeric)
    print(dataframe.dtypes)
    return dataframe


dfXtest = dftest[colz].iloc[:, 8:18]
dfXtest['Fare'] = dfXtest['Fare'].fillna(value='15')

dfYtest = dftest.iloc[:, 2]

# Testing for missing data


# -RANDOM FOREST

def Forest(Xtrain, Ytrain, Xtest, Ytest):

    # Train
    rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=69)
    rf.fit(Xtrain, Ytrain)

    # Feature Importance
    rfFeature_importances = pd.DataFrame(rf.feature_importances_, index=dfXtrain.columns,
                                         columns=['importance']).sort_values('importance', ascending=False)

    # Test
    rfPredictions = rf.predict(Xtest)
    rfAccuracy = accuracy_score(Ytest, rfPredictions)

    print(f'Baggie score: {rf.oob_score_:.3}')
    print(f'Mean Accuracy: {rfAccuracy:.3}')

    #Visualize Tree

    tree = rf.estimators_[999]
    export_graphviz(tree, out_file='tree.dot',
                    rounded=True, proportion=True,
                    precision=2, filled=True)

    image = Source.from_file("/Users/antonis/PycharmProjects/OD14/tree.dot")
    image = image.view()
    return rf, rfFeature_importances, rfPredictions, rfAccuracy, print(f'Mean Accuracy: {rfAccuracy:.3}'), image

# Visualizing TreeForest(dfXtrain, dfYtrain, dfXtest, dfYtest)


# -XGBoost


def XGBooster(Xtrain, Ytrain, Xtest, Ytest):

    # Train
    XG = XGBClassifier()
    XG.fit(Xtrain, Ytrain)

    #Feature Importancess
    XGFeature_importances = pd.DataFrame(XG.booster().get_fscore(), index=Xtrain.columns,
                                         columns=['importance']).sort_values('importance', ascending=False)

    # Test
    yPred = XG.predict(Xtest)
    XGPredictions = [round(value) for value in yPred]
    XGAccuracy = accuracy_score(Ytest, XGPredictions)
    return XGFeature_importances, XG, yPred, XGPredictions, XGAccuracy, print("Accuracy: %.2f%%" % (XGAccuracy * 100.0))


# Visualizing Tree




#RUN Randomtrees: Forest(dfXtrain, dfYtrain, dfXtest, dfYtest)
#RUN XGBOOST: XGBooster(dfXtrain, dfYtrain, dfXtest, dfYtest)





import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


os.chdir('/Users/antonis/Desktop/OdysseyProjects/OD13')

# 1st way to connect files:
#df = pd.concat(map(pd.read_csv,
                   #glob.glob(os.path.join("*e6fff10-b6db-4f7d-842d-30eda47ac10d-c000.csv",))))

# 2nd way to connect files (df = pd.concat([pd.read_csv(i, header = None) for i in list ignore_index = True]))
list = sorted(glob.glob("*e6fff10-b6db-4f7d-842d-30eda47ac10d-c000.csv"))

df = pd.DataFrame()

for i in list:
    df1 = pd.read_csv(i, header = None)
    df = df.append(df1, ignore_index=True)

df.columns = ["cdatetime", "src_username", "source", "destination", "sourceport", "service", "action"]

#Convert Datetime
df['cdatetime'] = pd.to_datetime(df['cdatetime'])

#Count entries per user
#dfCount = pd.value_counts(df.src_username).to_frame().reset_index()

#Set Options
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 10)

# Separating NAN values from rest of dataframe
dfNorm = df[df.src_username.notnull()].reset_index()
dfNan = df[df.src_username.isnull()].reset_index()

#How many: 1) Destination IP's, 2) Source Ports, 3) Destination Ports
dfDestIP = dfNorm.groupby(['src_username', pd.Grouper(key = 'cdatetime', freq = 'H' )])['destination'].nunique()
dfSource = dfNorm.groupby(['src_username', pd.Grouper(key = 'cdatetime', freq = 'H' )])['sourceport'].nunique
dfDestPort = dfNorm.groupby(['src_username', pd.Grouper(key = 'cdatetime', freq = 'H' )])['service'].nunique()

#Ratio accept:total
#a)Grouping Username with Source IP (hourly)
dfGroupNorm = dfNorm.groupby(['src_username', pd.Grouper(key = 'cdatetime', freq = 'H'),'action']).size().reset_index(drop = False)
dfGroupNorm = dfGroupNorm.rename({0:'freq'}, axis = 'columns')

dfGroupNormP = dfGroupNorm.pivot_table(index = ['src_username', 'cdatetime'], columns = 'action',  values = 'freq' )
dfGroupNormP = dfGroupNormP.reset_index().fillna(0)
#b) Adding and dividing to get ratio

cLength = len(dfGroupNormP['accept'])

dfGroupNormP['Total'] = dfGroupNormP['accept'] + dfGroupNormP['drop']
dfGroupNormP['Ratio'] = dfGroupNormP['accept']/dfGroupNormP['Total']
#dfGroupNormPF = dfGroupNormP.replace(0, pd.np.nan).reset_index()
#dfGroupNormPF = dfGroupNormPF[dfGroupNormPF.Ratio.notnull()]

#Plot Line Graph

#dfGroupNormPSort = dfGroupNormP.sort_values(by = ['cdatetime'], ascending = False)
#dfPlot1 = mdates.date2num(dfGroupNormP['cdatetime'])
#dfGraph = plt.plot(a['cdatetime'],a['Ratio'],'o').show()

#Function for plotting graph by name

namesList = df.src_username.unique()



def plotName(datas_Frame, name):
    graph = datas_Frame.loc[datas_Frame['src_username'] == name]
    plt.figure()
    plotting = plt.plot(graph['cdatetime'], graph['Ratio'], 'o')


z = 0

for username in namesList:
    plotName(dfGroupNormP, username)
    z += 1
    plt.savefig('/Users/antonis/Desktop/OdysseyProjects/OD13/Graphs/' + 'Graph_' + str(z))
    plt.close()










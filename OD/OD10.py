import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/antonis/Desktop/OdysseyProjects/OD1/query-impala-28596.csv")

dfremove = df.drop('cdatetime', axis = 1)

dftime = pd.to_datetime(df['cdatetime'])

df['cdatetime'] = dftime

df['cdatetime'] = pd.to_datetime(df['cdatetime'])

df1 = df.reset_index()

df2 = df1.groupby(['username', pd.Grouper(key = 'cdatetime', freq = 'H')])['id'].count()

df3 = df1.groupby(['username', pd.Grouper(key = 'cdatetime', freq = '15min')])['id'].count()

df4 = df1.groupby(['username', pd.Grouper(key = 'cdatetime', freq = '30min')])['id'].count()
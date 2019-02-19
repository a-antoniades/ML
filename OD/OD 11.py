import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/antonis/Desktop/OdysseyProjects/OD1/query-impala-28596.csv")

#ex.1 - find highest frequency of user with 4625

dfcnt = df.groupby(['username', 'eventid']).agg({'host':'count'})
dfcnt = dfcnt.reset_index()

d_4625 = dfcnt[dfcnt['eventid'] == 4625]


#ex.2 - Graph

dfcnt0 = df.groupby(['username']).agg({'host':'count'})
dfgr0 = dfcnt0.sort_values(['host'], ascending=False)
dfgr0_1 = dfgr0.reset_index()

users = dfgr0_1.iloc[0:5,0].tolist()
dfcnt2 = dfcnt[dfcnt['username'].isin(users)]
dfgr2 = dfcnt2.sort_values(['host', 'username', 'eventid'], ascending = False)

dfpv = pd.pivot_table(dfcnt, index=['username', 'eventid'])
dfgr = dfcnt.sort_values(['username', 'eventid'], ascending = True)

dfgr2pv = dfgr2.pivot('username', 'eventid', 'host', ).plot(kind = 'bar') #.sort_values(['eventid', 'username'])

end = plt.show()

dfgr2pv_1 = dfgr2pv.fillna(0)

dfgr2pv_1E = dfgr2pv_1.plot(kind = 'bar')

end = plt.show()









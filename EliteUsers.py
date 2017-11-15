'''
Created on Nov 12, 2017

@author: purbasha
'''
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

filename = 'VMShare/boardgame-elite-users.csv'
csvfile = open(filename, 'rU')
reco = {}
filereader = csv.reader(csvfile, delimiter=",")
next(filereader, None)  
st_lst = []
rec= {}

filegame = 'VMShare/boardgame-titles.csv'
csvgame = open(filegame, 'rU')
filereadergame= csv.reader(csvgame, delimiter=",")
next(filereadergame, None)

cnt = {}
for data in filereader:
    if data[1] not in rec:
        rec[data[1]]=float(data[2]) 
        cnt[data[1]]=1  
    else:
        rec[data[1]]+=float(data[2])
        cnt[data[1]]+=1 
       
avg={}

for k1, v1 in rec.iteritems():
    for k2,v2 in cnt.iteritems():
        if k1==k2:
            avg[k1]=float(v1)/float(v2)            

#print avg
lists = sorted(avg.items(), key=lambda x: x[1], reverse=True)        
lists = lists[:20]      

gameid = zip(*lists)[0]
gamename=[]
for id in filereadergame:
    for game in gameid:
        if id[0]==game:
            gamename.append(id[1])          

gamename = tuple(gamename)
score = zip(*lists)[1]
x_pos = np.arange(len(gamename)) 
# calculate slope and intercept for the linear trend line
slope, intercept = np.polyfit(x_pos, score, 1)
trendline = intercept + (slope * x_pos)

plt.plot(x_pos, trendline, color='red', linestyle='--')    
plt.bar(x_pos, score,align='center')
plt.xticks(x_pos, gamename, rotation=90) 
plt.ylabel('Popularity Score')
plt.gcf().subplots_adjust(bottom=0.30)
plt.show()

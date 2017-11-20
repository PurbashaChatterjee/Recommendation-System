import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


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
        rec[data[1]]=[float(data[2])]    
    else:
        rec[data[1]].append(float(data[2]))
       
new_rec={}            
for (key, val) in rec.iteritems():
    counter = Counter(val)   
    max_count = max(counter.values())
    mode = [k for k,v in counter.items() if v == max_count]
    new_rec[key] = mode[0]
    
lists = sorted(new_rec.items(), key=lambda x: x[1], reverse=True)
lists = lists[:20]      
        
gameid = zip(*lists)[0]
gamename=[]
for id in filereadergame:
    for game in gameid:
        if id[0]==game:
            #name = repr(id[1]).replace("\\", "")
            name = repr(id[1]).replace("\\x", "")
            gamename.append(name)          


gamename = tuple(gamename)
print gamename
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
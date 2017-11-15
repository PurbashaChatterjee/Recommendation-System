'''
Created on Nov 14, 2017

@author: purbasha
'''
import csv
import matplotlib.pyplot as plt
import numpy as np

filename = 'VMShare/boardgame-frequent-users.csv'
csvfile = open(filename, 'rU')
filereader = csv.reader(csvfile, delimiter=",")
next(filereader, None)  

filegame = 'VMShare/boardgame-titles.csv'
csvgame = open(filegame, 'rU')
filereadergame= csv.reader(csvgame, delimiter=",")
next(filereadergame, None)

def usersnum():
    cnt = {}
    for data in filereader:
        if data[1] not in cnt:
            cnt[data[1]]=1  
        else:
            cnt[data[1]]+=1 
    lists = sorted(cnt.items(), key=lambda x: x[1], reverse=True)        
    lists = lists[:20]      
    print lists
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

def highscores():   
    cnt = {}
    for data in filereader:
        if float(data[2]) > 9:
            if data[1] not in cnt:
                cnt[data[1]]=1  
            else:
                cnt[data[1]]+=1 
    lists = sorted(cnt.items(), key=lambda x: x[1], reverse=True)        
    lists = lists[:20]      
    print lists
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

#usersnum()
highscores()
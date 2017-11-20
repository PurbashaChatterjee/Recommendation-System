'''
Created on Nov 19, 2017

@author: purbasha
'''
import pandas as pd
from sklearn import svm
from sklearn import neural_network
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

filename= 'VMShare/boardgame-users.csv'
csvfile = open(filename, 'rU')
filereader = pd.read_csv(csvfile)
filereaderFreq=filereader.rename(columns = {"Compiled from boardgamegeek.com by Matt Borthwick":'userID'})

#clf = neural_network.MLPClassifier()
#clf = svm.SVR()
def usersnum():
    X_train = []
    Y_train = []
    j = 0
    X_test = []
    Y_test = []
    for (usr, gameid, rate) in zip(filereaderFreq.userID, filereaderFreq.gameID, filereaderFreq.rating):
        if float(rate) < 4:
            cls = 1
        elif float(rate) > 3 and float(rate)< 7:
            cls = 2
        else:
            cls=3          
        if j%4==0:
            X_test.append([usr, gameid])
            Y_test.append(cls)
        else:    
            X_train.append([usr, gameid])
            Y_train.append(cls)
        j+=1    
    print ("Let's train...")    
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train) 
    print ("Model trained") 
    kmeans.labels_
    #svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    #kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False) 
    print ("Let's predict")
    #y_pred = clf.predict(X_test)
    y_pred = kmeans.predict(X_test)
    print ("Done with the prediction")
    print(classification_report(Y_test, y_pred))
 
 
usersnum()      
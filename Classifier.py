'''
Created on Nov 19, 2017

@author: purbasha
'''
import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np
import keras
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop



filename= 'VMShare/boardgame-frequent-users.csv'
csvfile = open(filename, 'rU')
filereader = pd.read_csv(csvfile)
filereader=filereader.rename(columns = {"Compiled from boardgamegeek.com by Matt Borthwick":'userID'})

filename_elite= 'VMShare/boardgame-elite-users.csv'
csvfile_elite = open(filename_elite, 'rU')
filereader_elite = pd.read_csv(csvfile_elite)
filereader_elite=filereader_elite.rename(columns = {"Compiled from boardgamegeek.com by Matt Borthwick":'userID'})

def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01)

num_users, num_items = 0, 0
def usersnum(filereader_data): 
    usrGame = {}
    user_list = []   
    for (usr, gameid, rate) in zip(filereader_data.userID, filereader_data.gameID, filereader_data.rating):       
        if usr not in usrGame:
            usrGame[usr]={}
            user_list.append(usr)
        if gameid not in usrGame[usr]:
            usrGame[usr][gameid] = rate    
   
    df = pd.DataFrame.from_dict(v1 for k1,v1 in usrGame.iteritems())  
    df = df.fillna(0)
    df.head()
    
    mat = df.as_matrix()    
    rate_mean = np.mean(mat, axis=1)  
    normalize_mat = mat - rate_mean.reshape(-1,1)  
    user_mat, diagonal, game_mat = svds(normalize_mat, 50)
    diagonal = np.diag(diagonal)
    blank_pred_rate = np.dot(np.dot(user_mat, diagonal), game_mat) + rate_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(blank_pred_rate, columns = df.columns)
    preds_df["Users"] = user_list
    
    return preds_df


    
def deepModel(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(num_users,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(num_items,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = 100, output_dim = latent_dim, name = 'user_embedding')
    MF_Embedding_Game = Embedding(input_dim = 100, output_dim = latent_dim, name = 'item_embedding')   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Game(item_input))
    user_latent = Dense(100, activation='relu', kernel_regularizer='l2')(user_latent)
    item_latent = Dense(100, activation='relu', kernel_regularizer='l2')(item_latent)
    # Element-wise product of user and item embeddings 
    predict_vector = merge([user_latent, item_latent], mode = 'concat')
    
    predict_vector = Dense(200, activation = 'relu', kernel_regularizer='l2')(predict_vector)
    predict_vector = Dense(50, activation = 'relu', kernel_regularizer='l2')(predict_vector)
    prediction = Dense(1, activation='sigmoid', name = 'prediction')(predict_vector)
    
    model = Model(input=[user_input, item_input], output=prediction)

    return model 

def getTrain(train):
    train_input,game_input, labels = [],[],[]
    num_items = train.shape[1]
    column = train.columns.tolist()   
    for row in train.iterrows():  
        col1 = 0
        for rate in row[1]:
            if col1 < 402:
                train_input.append(row[1][num_items-1])
                game_input.append(column[col1])
                labels.append(rate)
                col1+=1                  
  
    return train_input,game_input, labels


pred_df= usersnum(filereader) 
val_pred = usersnum(filereader_elite)
data_train, data_game, labels = getTrain(pred_df) 
val_train, val_game, val_labels = getTrain(val_pred)  

#model = get_model(193504, 402, 5, [0.6,0.5])
#model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy')
model = deepModel(pred_df.shape[0], pred_df.shape[1], 10)

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    
model.fit([np.array(data_train),np.array(data_game)], labels,
          batch_size=100,
          epochs=10,
          verbose=1,
          validation_data=([np.array(val_train),np.array(val_game)], val_labels))
   

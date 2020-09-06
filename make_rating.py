import numpy as np
import tensorflow as tf
import os
import pandas as  pd
from data.dataset import _get_training_data, _get_test_data
from model.train_model import TrainModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import gc
from tqdm import tqdm
import h5py 

def build_vector(music_ids,count_list,music_vocab):
    vec=np.zeros(2400)
    for music,cnt in zip(music_ids,count_list):
        index=music_vocab[music]
        vec[index]=cnt
    vec=vec/vec.max()
    return vec

def get_music_data():
    ''' For each train.dat and test.dat making a User-Movie-Matrix'''
    
    gc.enable()
    
    path='data/labels.csv'
    data=pd.read_csv(path)

    user_ids=data['user_id'].unique()
    with open('data/music_ids.txt') as f:
        music_ids=f.readlines()
    
    music_ids=[item.strip() for item in music_ids]
    music_vocab={}
    for index,item in enumerate(music_ids):
        music_vocab[item]=index

    user_data=data.groupby('user_id')

    list_vec=[]
    cnt=0
    files=open('data/ratings.txt','w')
    for group in tqdm(user_data):
        # print(group)
        user_id=group[0]
        music_ids=group[1]['music_id'].tolist()
        count_list=group[1]['count'].tolist()
        # print(count_list)
        vec=build_vector(music_ids,count_list,music_vocab)

        count_vec=np.array(count_list)
        count_rating=count_vec/np.max(count_vec)
        
        list_vec.append(vec)
        for music_id,rating in zip(music_ids,count_rating):
            files.write(user_id+'\t'+music_id+'\t'+str(rating)+'\n')

        cnt+=1
        # if(cnt==1000):
        #     break

        del vec

    
    return list_vec


if __name__ == "__main__":
    get_music_data()
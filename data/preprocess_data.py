import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
from tqdm import tqdm 

p = Path(__file__).parents[1]

ROOT_DIR=os.path.abspath(os.path.join(p, '..', 'data/raw/'))

def convert(data, num_users, num_movies):
    ''' Making a User-Movie-Matrix'''
    
    new_data=[]
    
    for id_user in range(1, num_users+1):
        
        id_movie=data[:,1][data[:,0]==id_user]
        id_rating=data[:,2][data[:,0]==id_user]
        ratings=np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie-1]=id_rating
        if sum(ratings)==0:
            continue
        print(ratings.shape)
        print(ratings.tolist())
        new_data.append(ratings)

        del id_movie
        del id_rating
        del ratings
    # print(new_data) 
    print(len(new_data))
    return new_data

def get_dataset_1M():
    ''' For each train.dat and test.dat making a User-Movie-Matrix'''
    
    gc.enable()
    
    training_set=pd.read_csv(ROOT_DIR+'/ml-1m/train.dat', sep='::', header=None, engine='python', encoding='latin-1')
    training_set=np.array(training_set, dtype=np.uint32)
    
    test_set=pd.read_csv(ROOT_DIR+'/ml-1m/test.dat', sep='::', header=None, engine='python', encoding='latin-1')
    test_set=np.array(test_set, dtype=np.uint32)
    
      
    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))

    training_set=convert(training_set,num_users, num_movies)
    test_set=convert(test_set,num_users, num_movies)
    
    return training_set, test_set

def build_vector(music_ids,count_list,music_vocab):
    vec=np.zeros(2400)
    for music,cnt in zip(music_ids,count_list):
        index=music_vocab[music]
        vec[index]=cnt
    vec=vec/vec.max()
    # print(vec.tolist())
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
    for group in tqdm(user_data):
        music_ids=group[1]['music_id'].tolist()
        count_list=group[1]['count'].tolist()
        vec=build_vector(music_ids,count_list,music_vocab)
        list_vec.append(vec)

        del vec
    
    return list_vec


def _get_dataset():

    return get_dataset_1M()

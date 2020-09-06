import h5py
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 
import numpy as np


musics=h5py.File('data/music_latent_vector.h5')

users=h5py.File('data/user_latent_vector.h5')
file=open('data/score_list.txt','w')
recommend_score=h5py.File('data/recommend_score.h5')
for user_id,user_value in tqdm(users.items()):
    list_vec=[]
    for music_id,music_value in musics.items():
        # print(user_value.shape)
        res=cosine_similarity(user_value[:].reshape(1,-1),music_value[:].reshape(1,-1))
        list_vec.append(res[0][0])
        # file.write(user_id+'\t'+music_id+'\t'+str(res[0][0])+'\n')
        # print(res)
        # break
    list_vec=np.array(list_vec)
    recommend_score.create_dataset(user_id,data=list_vec)
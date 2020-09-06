import h5py
from generate_music_tfrecord import read_tfRecord,load_train_lyrics
import numpy as np
from average_precision import apk,mapk
import glob
from tqdm import tqdm 

# root_path='/media/data/data/MillionSongSubset/data/*/*/*/*.h5'
# h5files=glob.glob(root_path)
# dict_lyrics=load_train_lyrics()
# index=0
# positions=[]
# for h5file in tqdm(h5files):
#     hdfFile = h5py.File(h5file, 'r')
#     track_id=hdfFile['analysis']['songs']['track_id'][0].decode('UTF-8')
#     if(track_id in dict_lyrics):
#         positions.append(index)
#     index+=1



with open('data/music_ids.txt') as f:
    music_ids=f.readlines()

music_ids=[item.strip() for item in music_ids]

music_vocab={}
for index,item in enumerate(music_ids):
    music_vocab[item]=index

user_label_vector=h5py.File('data/user_label_vector.h5')

recommend_score=h5py.File('data/recommend_score.h5')
list_gt=[]
list_pred=[]
for (user_id,user_value),(_,value_score) in tqdm(zip(user_label_vector.items(),recommend_score.items())):
    # user_vec=[user_value[index] for index in positions]
    user_vec=user_value[:]
    user_vec=np.argsort(user_vec)[-500:]
    # print(value_score)
    value_vec=np.argsort(value_score[:])[-500:]
    # print(value_vec)
    # print(user_vec)
    value_vec=value_vec.tolist()
    value_vec.reverse()

    user_vec=user_vec.tolist()
    user_vec.reverse()
    list_pred.append(value_vec)
    list_gt.append(user_vec)

res500=mapk(list_gt,list_pred, 500)
res100=mapk(list_gt,list_pred, 100)
res50=mapk(list_gt,list_pred, 50)
print('map@500:{}'.format(res500))
print('map@100:{}'.format(res100))
print('map@50:{}'.format(res50))
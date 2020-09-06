import glob
import numpy as np
import h5py
import os
from tqdm import tqdm
from generate_music_tfrecord import load_train_lyrics

root_path='/media/data/data/MillionSongSubset/data/*/*/*/*.h5'

dict_lyrics=load_train_lyrics()
h5files=glob.glob(root_path)
list_ids=[]
for h5file in tqdm(h5files):
    hdfFile = h5py.File(h5file, 'r')
    song_id=hdfFile['metadata']['songs']['song_id']
    # print(song_id)
    key=h5file.split('/')[-1]
    key=key[:-3]
    if(key in dict_lyrics):
        list_ids.append(song_id[0].decode('UTF-8'))
    # segments_pitches=hdfFile['analysis']['segments_pitches'][:]
    # segments_timbre=hdfFile['analysis']['segments_timbre'][:]
    # segments_pitches=segments_pitches[:120,:]
    # segments_timbre=segments_timbre[:120,:]

    # segments_pitches=segments_pitches.transpose(1,0)
    # segments_timbre=segments_timbre.transpose(1,0)

    # segments_pitches=np.expand_dims(segments_pitches,axis=2)
    # segments_timbre=np.expand_dims(segments_timbre,axis=2)
    # x=np.concatenate((segments_pitches,segments_timbre),axis=2)
    # print(x.shape)
    # md5=hdfFile['analysis']['songs']['audio_md5']
    # print(md5)
    # break
# list_ids=[]
# for h5file in h5files:
#     music_id=h5file.split('/')[-1]
#     music_id=music_id[:-3]
#     list_ids.append(music_id)

with open('data/music_ids.txt','w') as f:
    for item in list_ids:
        f.write(item+'\n')

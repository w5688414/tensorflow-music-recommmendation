import pandas as pd


path='/media/data/data/train_triplets.txt'
data=pd.read_csv(path,sep='\t',header=None)

with open('data/music_ids.txt') as f:
    music_ids=f.readlines()

music_ids=[item.strip() for item in music_ids]


data.rename(columns={0:'user_id',1:'music_id',2:'count'},inplace=True)
# print(data.head())
# print(music_ids)
music_data=data[data['music_id'].isin(music_ids)]
music_data.to_csv('data/labels.csv',index=False)
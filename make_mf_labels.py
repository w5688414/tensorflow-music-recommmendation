import pandas as pd

data=pd.read_csv('data/labels.csv')

with open('data/music_ids.txt') as f:
    music_ids=f.readlines()
    
music_ids=[item.strip() for item in music_ids]
music_vocab={}
for index,item in enumerate(music_ids):
    music_vocab[item]=index

with open('data/user_ids.txt') as f:
    user_ids=f.readlines()

user_ids=[item.strip() for item in user_ids]
user_vocab={}
for index,item in enumerate(user_ids):
    user_vocab[item]=index

data['user_id']=data['user_id'].apply(lambda x:user_vocab[x])
data['music_id']=data['music_id'].apply(lambda x:music_vocab[x])

data.to_csv('data/ratings.csv',index=False)

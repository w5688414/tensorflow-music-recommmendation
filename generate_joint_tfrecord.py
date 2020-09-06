
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm  
import glob
import h5py
import pandas as pd

def load_file(root_path):
    with open(root_path,'r') as f:
        data=f.readlines()
    return data


def load_h5files(path):
    hdfile=h5py.File(path)
    return hdfile

def parse(lyrics,word_vec,id_word):
    list_vec=[]
    for item in lyrics:
        index=item.split(':')[0]
        index=int(index)
        word=id_word[index]
        vec=word_vec[word]
        # print(type(vec))
        list_vec.append(vec)
    vectors=np.array(list_vec)
    vectors=np.sum(vectors, axis=0)
    return vectors
    

def train2tfRecord(output_dir,_examples):
    
    filename = output_dir + '.tfrecords'
    print(len(_examples))
    music_input=load_h5files('data/music_input.h5')
    user_input=load_h5files('data/user_label_vector.h5')
    word_vec_input=load_h5files('data/music_wv.h5')
    cnt=0
    writer = tf.python_io.TFRecordWriter(filename)
    for i,example in tqdm(enumerate(_examples)):
        arr=example.strip().split('\t')
        user_id=arr[0]
        track_id=arr[1]
        rating=arr[2]
        
        uesr_vec=user_input.get(user_id)[:]
        music_vec=music_input.get(track_id)
        word_vec=word_vec_input.get(track_id)
        # image_raw = image.tostring()
        
        if(music_vec is None):
            print(track_id)
            print(user_id)
            print(uesr_vec)
            print(music_vec)
            cnt+=1
            continue
        music_vec=music_vec[:].flatten() #这里一定要铺平，不然存不进去
        word_vec=word_vec[:].flatten()
        example = tf.train.Example(features=tf.train.Features(feature={
                'user_vec':tf.train.Feature(float_list=tf.train.FloatList(value=uesr_vec)),
                'music_vec':tf.train.Feature(float_list=tf.train.FloatList(value=music_vec)),
                'word_vec':tf.train.Feature(float_list=tf.train.FloatList(value=word_vec)),
                'rating':tf.train.Feature(float_list=tf.train.FloatList(value=[float(rating)]))                       
                }))
        writer.write(example.SerializeToString())
    writer.close()
    print(cnt)
    return filename



def decode(serialized_example):
    features=tf.parse_single_example(serialized_example,
                                    features={
                                        'user_vec':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
                                        'music_vec':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
                                        'word_vec':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
                                        'rating':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True)
                                    })
    image = tf.cast(features['music_vec'],tf.float32)
    user_vec=tf.cast(features['user_vec'],tf.float32)
    word_vec=tf.cast(features['word_vec'],tf.float32)
    rating=tf.cast(features['rating'],tf.float32)

    music_vec = tf.reshape(image,[12,120,2])
    user_vec=tf.reshape(user_vec,[2400])
    word_vec = tf.reshape(word_vec,[500])

    music_vec = tf.cast(music_vec, tf.float32)

    return music_vec,user_vec,word_vec,rating

def read_tfRecord(file_tfRecord,batch_size):
    dataset=tf.data.TFRecordDataset(file_tfRecord)
    dataset=dataset.map(decode)
    dataset=dataset.batch(batch_size)
    dataset=dataset.repeat()
    iterator=dataset.make_one_shot_iterator()
    
    return iterator


if __name__ == "__main__":
    root_path='data/ratings.txt'
    output_dir='data/train_joint'
    _examples = load_file(root_path)
    sep_num=300000
    train_files=_examples[:sep_num]
    train2tfRecord(output_dir,train_files)
    trainroad='data/train_joint.tfrecords'
    iterator=read_tfRecord(trainroad)

    output_dir='data/test_joint'
    test_files=_examples[sep_num:]
    train2tfRecord(output_dir,test_files)
    trainroad='data/test_joint.tfrecords'
    iterator=read_tfRecord(trainroad)

    with tf.Session() as sess:
        for i in range(100):
            music_vec,user_vec,word_vec,rating=iterator.get_next()
            music_vec,user_vec,word_vec,rating=sess.run([music_vec,user_vec,word_vec,rating])
            music_vec=np.array(music_vec)
            music_vec=np.array(music_vec)
            print(music_vec.shape)
            # print(vector_batch.shape)
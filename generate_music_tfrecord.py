
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm  
import glob
import h5py
import pandas as pd

def load_file(root_path):
    h5files=glob.glob(root_path)
    return h5files

def load_txt(txt_path):
    with open(txt_path) as f:
        data=f.readlines()
    return data

def load_train_lyrics():
    train_txt_path='/media/data/data/mxm_dataset_train.txt'
    data=[]
    data+=load_txt(train_txt_path)[18:]
    dict_lyrics={}
    for item in data:
        item_arr=item.strip().split(',')
        music_id=item_arr[0]
        mxm_id=item_arr[1]
        word_count=item_arr[2:]
        dict_lyrics[music_id]=word_count
    return dict_lyrics

def load_test_lyrics():
    data=[]
    test_txt_path='/media/data/data/mxm_dataset_test.txt'
    data+=load_txt(test_txt_path)[18:]
    dict_lyrics={}
    for item in data:
        item_arr=item.strip().split(',')
        music_id=item_arr[0]
        mxm_id=item_arr[1]
        word_count=item_arr[2:]
        dict_lyrics[music_id]=word_count
    return dict_lyrics

def load_word_vec():
    data=pd.read_csv('data/word_to_vec.csv')
    words=data['word'].tolist()
    vectors=np.load('data/vec.npy')
    indexes=data['index'].tolist()
    id_word={}
    word_vec={}
    for word,index,vec in zip(words,indexes,vectors):
        word_vec[word]=vec
        id_word[index]=word
    return word_vec,id_word

def load_tfidf(path):
    doc_vec={}
    with open(path) as f:
        data=f.readlines()
        for line in data:
            # print(line)
            line_arr=line.strip().split('\t')
            if(len(line_arr)<2):
                continue
            doc_id,value=line_arr
            doc_vec[doc_id]=value

    return doc_vec
    
def word_tfidf(tfidfs):
    tfidf_word={}
    list_tfidf=tfidfs.split()
    for item in list_tfidf:
        word,value=item.split(':')
        tfidf_word[word]=float(value)
    return tfidf_word


def parse(lyrics,tfidfs,word_vec,id_word):
    tfidf_word=word_tfidf(tfidfs)
    list_vec=[]
    total_count=0
    for item in lyrics:
        index,count=item.split(':')
        index=int(index)
        count=int(count)
        word=id_word[index]
        vec=word_vec[word]
        if(word not in tfidf_word):
            print(word)
            continue
        tfidf=tfidf_word[word]  # get tfidf
        total_count+=count
        vec=tfidf*vec
        
        list_vec.append(vec)
    vectors=np.array(list_vec)
    vectors=np.sum(vectors, axis=0)/total_count
    return vectors
    
def test2tfRecord(_examples,output_dir):
    # _examples = load_file(trainFile)
    filename = output_dir + '.tfrecords'
    dict_lyrics=load_train_lyrics()
    word_vec,id_word=load_word_vec()
    tfidf_vocab=load_tfidf('data/tfidf_train.txt')
    cnt=0
    writer = tf.python_io.TFRecordWriter(filename)
    for i,example in tqdm(enumerate(_examples)):
        key=example.split('/')[-1]
        key=key[:-3]
        # print(key)
        if(key in dict_lyrics):
            cnt+=1
            lyrics=dict_lyrics[key]
            tfidfs=tfidf_vocab[key]
            vector=parse(lyrics,tfidfs,word_vec,id_word)
        else:
            continue
            
        hdfFile = h5py.File(example, 'r')
        segments_pitches=hdfFile['analysis']['segments_pitches'][:]
        segments_timbre=hdfFile['analysis']['segments_timbre'][:]
        while(segments_timbre.shape[0]<120):
            segments_timbre=np.concatenate((segments_timbre,segments_timbre),axis=0)
            segments_pitches=np.concatenate((segments_pitches,segments_pitches),axis=0)
            print(segments_timbre.shape)
        segments_pitches=segments_pitches[:120,:]
        segments_timbre=segments_timbre[:120,:]
        if(segments_timbre.shape[0]<120 or segments_pitches.shape[0]<120 ):
            print(example)
            continue

        segments_pitches=segments_pitches.transpose(1,0)
        segments_timbre=segments_timbre.transpose(1,0)

        segments_pitches=np.expand_dims(segments_pitches,axis=2)
        segments_timbre=np.expand_dims(segments_timbre,axis=2)
        image=np.concatenate((segments_pitches,segments_timbre),axis=2)
        # print(image.shape)

        # image_raw = image.tostring()
        image_raw=image.flatten() #这里一定要铺平，不然存不进去
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':tf.train.Feature(float_list=tf.train.FloatList(value=image_raw)),
                'word_vec':tf.train.Feature(float_list=tf.train.FloatList(value=vector))                       
                }))
        writer.write(example.SerializeToString())
    writer.close()
    print(cnt)
    return filename



def train2tfRecord(_examples,output_dir):
    
    filename = output_dir + '.tfrecords'
    print(len(_examples))
    dict_lyrics=load_train_lyrics()
    tfidf_vocab=load_tfidf('data/tfidf_train.txt')
    word_vec,id_word=load_word_vec()
    cnt=0
    writer = tf.python_io.TFRecordWriter(filename)
    for i,example in tqdm(enumerate(_examples)):
        key=example.split('/')[-1]
        key=key[:-3]
        # print(key)
        if(key in dict_lyrics):
            cnt+=1
            lyrics=dict_lyrics[key]
            tfidfs=tfidf_vocab[key]
            vector=parse(lyrics,tfidfs,word_vec,id_word)
        else:
            # print(key)
            continue
            
        hdfFile = h5py.File(example, 'r')
        segments_pitches=hdfFile['analysis']['segments_pitches'][:]
        segments_timbre=hdfFile['analysis']['segments_timbre'][:]
        while(segments_timbre.shape[0]<120):
            segments_timbre=np.concatenate((segments_timbre,segments_timbre),axis=0)
            segments_pitches=np.concatenate((segments_pitches,segments_pitches),axis=0)
            print(segments_timbre.shape)
        segments_pitches=segments_pitches[:120,:]
        segments_timbre=segments_timbre[:120,:]
        if(segments_timbre.shape[0]<120 or segments_pitches.shape[0]<120 ):
            print(example)
            continue

        segments_pitches=segments_pitches.transpose(1,0)
        segments_timbre=segments_timbre.transpose(1,0)

        segments_pitches=np.expand_dims(segments_pitches,axis=2)
        segments_timbre=np.expand_dims(segments_timbre,axis=2)
        image=np.concatenate((segments_pitches,segments_timbre),axis=2)
        # print(image.shape)

        # image_raw = image.tostring()
        image_raw=image.flatten() #这里一定要铺平，不然存不进去
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':tf.train.Feature(float_list=tf.train.FloatList(value=image_raw)),
                'word_vec':tf.train.Feature(float_list=tf.train.FloatList(value=vector))                       
                }))
        writer.write(example.SerializeToString())
    writer.close()
    print(cnt)
    return filename

def decode(serialized_example):
    features=tf.parse_single_example(serialized_example,
                                    features={
                                        'image_raw':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
                                        'word_vec':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True)
                                    })
    image = tf.cast(features['image_raw'],tf.float32)
    word_vec=tf.cast(features['word_vec'],tf.float32)
    image = tf.reshape(image,[12,120,2])
    word_vec = tf.reshape(word_vec,[500])
    return image,word_vec

def read_tfRecord(file_tfRecord):
    dataset=tf.data.TFRecordDataset(file_tfRecord)
    dataset=dataset.map(decode)
    batch_size=4
    dataset=dataset.batch(batch_size)
    dataset=dataset.repeat()
    iterator=dataset.make_one_shot_iterator()
    
    return iterator


if __name__ == "__main__":
    root_path='/media/data/data/MillionSongSubset/data/*/*/*/*.h5'
    output_dir='data/music_train'
    _examples = load_file(root_path)
    # train_files=_examples[:8000]
    train_files=_examples
    train2tfRecord(train_files,output_dir)
    trainroad=output_dir+'.tfrecords'
    iterator=read_tfRecord(trainroad)

    # output_dir='data/music_test'
    # test_files=_examples[8000:]
    # test2tfRecord(test_files,output_dir)
    # trainroad=output_dir+'.tfrecords'
    # iterator=read_tfRecord(trainroad)

    with tf.Session() as sess:
        for i in range(100):
            image_batch,vector_batch=iterator.get_next()
            image_batch,vector_batch=sess.run([image_batch,vector_batch])
            image_batch=np.array(image_batch)
            vector_batch=np.array(vector_batch)
            # print(image_batch.shape)
            # print(vector_batch.shape)
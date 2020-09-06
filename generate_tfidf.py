# coding:utf-8

import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py 

def load_h5files(path):
    hdfile=h5py.File(path)
    return hdfile

def id_word_dict(words):
    list_words=words.split(',')
    id_to_word={}
    word_to_id={}
    writer=open('data/mxm_dataset_vocab.txt','w')
    for index, word in enumerate(list_words,1):
        id_to_word[index]=word
        word_to_id[word]=index
        writer.write(word+' '+str(index)+'\n')
        
    return id_to_word,word_to_id


def get_words_train():
    txt_path='/media/data/data/mxm_dataset_train.txt'
    with open(txt_path) as f:
        data=f.readlines()

    print(data[10:20])
    words=data[17]
    id_to_word,word_to_id=id_word_dict(words)

    list_words=[]
    list_ids=[]
    for item in tqdm(data[18:]):
        item_arr=item.strip().split(',')
        music_id=item_arr[0]
        mxm_id=item_arr[1]
        word_count=item_arr[2:]
        words=[]
        for item in word_count:
            arr=item.split(':')
            word=id_to_word[int(arr[0])]
            count=int(arr[1])
            for i in range(count):
                words.append(word)
        list_words.append(' '.join(words))
        list_ids.append(music_id)
    return list_words,list_ids

def get_words_test():
    txt_path='/media/data/data/mxm_dataset_test.txt'
    with open(txt_path) as f:
        data=f.readlines()

    print(data[10:20])
    words=data[17]
    id_to_word,word_to_id=id_word_dict(words)

    list_words=[]
    list_ids=[]
    for item in tqdm(data[18:]):
        item_arr=item.strip().split(',')
        music_id=item_arr[0]
        mxm_id=item_arr[1]
        word_count=item_arr[2:]
        words=[]
        for item in word_count:
            arr=item.split(':')
            word=id_to_word[int(arr[0])]
            count=int(arr[1])
            for i in range(count):
                words.append(word)
        list_words.append(' '.join(words))
        list_ids.append(music_id)
    return list_words,list_ids

def generate_train_tfidf(list_ids,tfIdf,list_words):
    output=open('data/tfidf_train.txt','w')
    for music_id,tf_idf_value in tqdm(zip(list_ids,tfIdf)):
        # print(type(tf_idf_value))
        Mc=tf_idf_value.tocoo()
        dict_data= {k:v for k,v in zip(Mc.col, Mc.data)}
        output.write(music_id+'\t')
        for col,v in dict_data.items():
            word=list_words[col]
            output.write(word+':'+str(v)+' ')
        output.write('\n')

def generate_test_tfidf(list_ids,tfIdf):
    output=open('data/tfidf_test.txt','w')
    for music_id,tf_idf_value in tqdm(zip(list_ids,tfIdf)):
        # print(type(tf_idf_value))
        Mc=tf_idf_value.tocoo()
        dict_data= {k:v for k,v in zip(Mc.col, Mc.data)}
        output.write(music_id+'\t')
        for k,v in dict_data.items():
            output.write(str(k)+':'+str(v)+' ')
        output.write('\n')
 
if __name__ == "__main__":
    corpus,list_ids=get_words_train()
    print(corpus[:10])

    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(corpus)
    list_words=tfIdfVectorizer.get_feature_names()
    generate_train_tfidf(list_ids,tfIdf,list_words)

    # data={'words':tfIdfVectorizer.get_feature_names(),'TF-IDF':tfIdf[0].T.todense().tolist()}
    # df=pd.DataFrame(list(data.items()),columns=['words', 'TF-IDF'])

    # df = pd.DataFrame()
    # df['words'] = tfIdfVectorizer.get_feature_names()
    # df['TF-IDF'] =tfIdf[0].T.todense()
    # df = df.sort_values('TF-IDF', ascending=False)
    # print (df.head(25))
    # df.to_csv('data/tfidf.csv',index=False)


    # print(tfIdf.toarray().shape)
    # print(len(corpus))
    
    # corpus_test,list_ids_test=get_words_test()
    # tfIdf_test = tfIdfVectorizer.fit_transform(corpus_test)
    # generate_test_tfidf(list_ids_test,tfIdf_test)


    # df = pd.DataFrame()
    # df['words'] = corpus
    # df['TF-IDF'] =tfIdf.toarray()
    # df = df.sort_values('TF-IDF', ascending=False)
    # print (df.head(25))
    # df.to_csv('data/tfidf.csv',index=False)
    

    # tf_idf_arr=tfIdf.toarray()
    writer=open('data/tfidf_vocab.txt','w')
    for index,word in enumerate(tfIdfVectorizer.get_feature_names(),1):
        writer.write(word+' '+str(index)+'\n')

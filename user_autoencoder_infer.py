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
    output=h5py.File('data/user_label_vector.h5')
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

        output.create_dataset(user_id,data=vec)
        cnt+=1
        # if(cnt==1000):
        #     break

        del vec

    
    return list_vec


class BaseModel(object):
        
    def __init__(self):
        
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.bias_initializer=tf.zeros_initializer()
        self._init_parameters()
    
    def _init_parameters(self):
        
        with tf.name_scope('weights'):
            self.W_1=tf.get_variable(name='weight_1', shape=(2400,2048), 
                                     initializer=self.weight_initializer)
            self.W_2=tf.get_variable(name='weight_2', shape=(2048,1940), 
                                     initializer=self.weight_initializer)
            self.W_3=tf.get_variable(name='weight_3', shape=(1940,2048), 
                                     initializer=self.weight_initializer)
            self.W_4=tf.get_variable(name='weight_4', shape=(2048,2400), 
                                     initializer=self.weight_initializer)
        
        with tf.name_scope('biases'):
            self.b1=tf.get_variable(name='bias_1', shape=(2048), 
                                    initializer=self.bias_initializer)
            self.b2=tf.get_variable(name='bias_2', shape=(1940), 
                                    initializer=self.bias_initializer)
            self.b3=tf.get_variable(name='bias_3', shape=(2048), 
                                    initializer=self.bias_initializer)
    
    def inference(self, x):
        ''' Making one forward pass. Predicting the networks outputs.
        @param x: input ratings
        
        @return : networks predictions
        '''
        
        with tf.name_scope('inference'):
             a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
             a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, self.W_2),self.b2))
             print(a2)
             a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, self.W_3),self.b3))   
             a4=tf.matmul(a3, self.W_4) 
        return a4

def main():
    '''Building the graph, opening of a session and starting the training od the neural network.'''
    checkpoints_path='../checkpoints'
    inference_graph=tf.Graph()
    output=h5py.File('user_vector.h5')
    with inference_graph.as_default():

        
        model=BaseModel()
        input_data=tf.placeholder(tf.float32, shape=[None, 2400])  
        ratings=model.inference(input_data)
        saver=tf.train.Saver()
        
        with tf.Session(graph=inference_graph) as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoints_path)   
            saver.restore(sess, ckpt.model_checkpoint_path)
            fc1=tf.get_default_graph().get_tensor_by_name('inference/Sigmoid_1:0')
            h5file='data/user_label_vector.h5'
            hdfFile = h5py.File(h5file, 'r')
            for user_id,v in hdfFile.items():
                # print(k,v)
                print(v.shape)
                print(user_id)
                v=np.expand_dims(v,axis=0)
                fc1_value = sess.run([fc1], feed_dict={input_data: v})
                fc1_value=np.array(fc1_value)
                print(fc1_value.shape)
                output.create_dataset(user_id,data=fc1_value[0])
                # break
            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)
                    
if __name__ == "__main__":
    get_music_data()
    # main()
    


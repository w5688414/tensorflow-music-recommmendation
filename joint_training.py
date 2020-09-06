import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from generate_joint_tfrecord import read_tfRecord
from tqdm import tqdm
import h5py 

LEARNING_RATE = 0.0005
BATCH_SIZE = 500
TS_BATCH_SIZE = 1000
N_EPOCHS = 20
REG_PENALTY = 0.05



##############################################################################################################################
################################################## ------ START HERE --------  ###############################################
##############################################################################################################################


def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.name_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out

def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out


def fullyConnected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        return out

def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out


def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs= n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                 activation_fn=tf.nn.relu)
        return out

def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


def music_encoder(music_batch,word_vec_batch, name):
    with tf.name_scope(name):
        c1 = conv2d(music_batch, name='c1', kshape=[12, 120, 2, 16])
        p1 = maxpool2d(c1, name='p1')
        print(p1)
        do1 = dropout(p1, name='do1', keep_rate=0.8)
        c2 = conv2d(do1, name='c2', kshape=[6, 60, 16, 16])
        p2 = maxpool2d(c2, name='p2')
        print(p2)
        do2 = tf.reshape(p2, shape=[-1, 1440])
        print(do2)
        dot3=tf.concat([do2, word_vec_batch], 1)
        fc1 = fullyConnected(dot3, name='fc1', output_size=500)
        print(fc1)


        # decoder
        fc2 = fullyConnected(fc1, name='fc2', output_size=1940)
        print(fc2)
        do4 = fc2[:,:1440]
        do4 = tf.reshape(do4, shape=[-1,1440,1,1])
        up1 = tf.image.resize_nearest_neighbor(do4, (6,60*16))
        up1 = tf.reshape(up1, shape=[-1,6,60,16])
        print(up1)
        d2 = conv2d(up1, name='d2', kshape=[6,60,16,16])
        up2 = tf.image.resize_nearest_neighbor(d2, (12,120))
        re3 = tf.reshape(up2, shape=[-1,12,120,16])
        output = conv2d(re3, name='d1', kshape=[12, 120,16,2])
    return fc2,output

def user_encoder(user_batch):
    bias_initializer=tf.zeros_initializer()
    weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2)
    with tf.name_scope('weights'):
        W_1=tf.get_variable(name='weight_1', shape=(2400,2048), 
                                     initializer=weight_initializer)
        W_2=tf.get_variable(name='weight_2', shape=(2048,1940), 
                                     initializer=weight_initializer)
        W_3=tf.get_variable(name='weight_3', shape=(1940,2048), 
                                     initializer=weight_initializer)
        W_4=tf.get_variable(name='weight_4', shape=(2048,2400), 
                                     initializer=weight_initializer)

    
    with tf.name_scope('biases'):
        b1=tf.get_variable(name='bias_1', shape=(2048), 
                                    initializer=bias_initializer)
        b2=tf.get_variable(name='bias_2', shape=(1940), 
                                    initializer=bias_initializer)
        b3=tf.get_variable(name='bias_3', shape=(2048), 
                                    initializer=bias_initializer)

        # with tf.name_scope('cost'):
        #     cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))


    with tf.name_scope('inference'):
        a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(user_batch, W_1),b1))
        a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, W_2),b2))
        a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, W_3),b3))   
        user_output=tf.matmul(a3, W_4)

    return a2,user_output



def CollabFilterring(user_batch,music_batch,word_vec_batch, name):
    print(music_batch)
    
    music,music_output=music_encoder(music_batch,word_vec_batch, name)
    music_variables = [var for var in tf.all_variables()]
    print(music_variables)
    user,user_output=user_encoder(user_batch)
    user_variables = [var for var in tf.all_variables() if(var not in music_variables)]
    print(user_variables)

    output = tf.reduce_sum(tf.multiply(music, user), 1, name='output')
	# output = tf.add(output, bias)
	# output = tf.add(output, batch_bias_movie)
	# output = tf.add(output, batch_bias_user, name='output')

    return music_variables,user_variables,output



def load_file(root_path):
    with open(root_path,'r') as f:
        data=f.readlines()
    return data

def load_h5files(path):
    hdfile=h5py.File(path)
    return hdfile
    

def train_nn_mf(user_batch, music_batch,word_vec_batch, rating_batch):
    NUM_TR_ROW=300000
    num_batch_loop = int(NUM_TR_ROW/BATCH_SIZE)
    music_variables,user_variables,prediction=CollabFilterring(user_batch, music_batch,word_vec_batch,'ConvAutoEnc')
    cost_l2 = tf.nn.l2_loss(tf.subtract(prediction, rating_batch))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_l2)
    saver = tf.train.Saver()
    trainroad='data/train_joint.tfrecords'
    iterator=read_tfRecord(trainroad,BATCH_SIZE)
    # tf.scalar_summary("cost_l2", cost_l2)

    test_path='data/train_joint.tfrecords'
    test_iterator=read_tfRecord(test_path,BATCH_SIZE)
    NUM_TEST_ROW=47320
    #构建这部分参数的
    saver1 = tf.train.Saver(music_variables)
    saver2=tf.train.Saver(user_variables)

    print('loading: '+str(music_variables))
    print('loading: '+str(user_variables))
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        
        music_checkpoints_path='music_checkpoint'
        ckpt = tf.train.get_checkpoint_state(music_checkpoints_path)
        print(ckpt)   
        saver1.restore(sess, ckpt.model_checkpoint_path)

        user_checkpoints_path='../checkpoints'
        ckpt = tf.train.get_checkpoint_state(user_checkpoints_path)
        print(ckpt)   
        saver2.restore(sess, ckpt.model_checkpoint_path)

     
    
        RMSEtr = []
        RMSEts=[]
        next_test_element=test_iterator.get_next()
        next_element=iterator.get_next()
        for epoch in range(N_EPOCHS):
            stime = time.time()
            num_batch_loop = int(NUM_TR_ROW/BATCH_SIZE)
            errors = deque(maxlen=num_batch_loop)
            TR_epoch_loss=0
            
            for i in tqdm(range(num_batch_loop)):
                music_vec,user_vec,word_vec,rating=sess.run(next_element)

                music_vec=np.array(music_vec)
                user_vec=np.array(user_vec)
                word_vec=np.array(word_vec)
                rating=np.array(rating)
                # print(music_vec.shape)
                # print(user_vec.shape)
                _, c, pred_batch = sess.run([optimizer, cost_l2, prediction], feed_dict = {user_batch: user_vec, music_batch: music_vec, word_vec_batch:word_vec,rating_batch:rating})
                errors.append(np.mean(np.power(pred_batch - rating, 2)))

            TR_epoch_loss = np.sqrt(np.mean(errors))
            RMSEtr.append(TR_epoch_loss)
            
            print("Epoch "+ str(epoch+1)+" completed out of "+str(N_EPOCHS)+"; Train loss:"+str(round(TR_epoch_loss,3)))
            saver.save(sess, "joint/model")

            num_eval_loop=int(NUM_TEST_ROW/BATCH_SIZE)
            test_errors=[]
            Test_epoch_loss=0
            
            for i in tqdm(range(num_eval_loop)):
                music_vec,user_vec,word_vec,rating=sess.run(next_test_element)

                music_vec=np.array(music_vec)
                user_vec=np.array(user_vec)
                word_vec=np.array(word_vec)
                rating=np.array(rating)

                c, pred_batch = sess.run([ cost_l2, prediction], feed_dict = {user_batch: user_vec, music_batch: music_vec, word_vec_batch:word_vec,rating_batch:rating})
                test_errors.append(np.mean(np.power(pred_batch - rating, 2)))
            Test_epoch_loss = np.sqrt(np.mean(test_errors))
            RMSEts.append(Test_epoch_loss)


            print("Validation "+ str(epoch+1)+" completed out of "+str(N_EPOCHS)+"; Validation loss:"+str(round(Test_epoch_loss,3)))


            

        plt.plot(RMSEtr, label='Training Set', color='b')
        plt.plot(RMSEts, label='Test Set', color='r')
        plt.legend()
        plt.ylabel('-----  RMSE  ---->')
        plt.xlabel('-----  Epoch  ---->')
        plt.title('RMSE vs Epoch (Biased Matrix Factorization)')
        plt.savefig('train.png')
        plt.show()
        print("Awesome !!")


def inference():
    trainFile='data/ratings.txt'
    _examples = load_file(trainFile)
    _examples=_examples[300000:]
    music_input=load_h5files('data/music_input.h5')
    user_input=load_h5files('data/user_label_vector.h5')
    word_vec_input=load_h5files('data/music_wv.h5')
    inference_graph=tf.Graph()
    cnt=0
    output_user=h5py.File('data/user_latent_vector.h5')
    output_music=h5py.File('data/music_latent_vector.h5')
    with inference_graph.as_default():
        user_batch = tf.placeholder(tf.float32, [None,2400], name='user_batch')
        music_batch = tf.placeholder(tf.float32, [None, 12,120,2], name='music_batch')
        word_vec_batch = tf.placeholder(tf.float32, [None, 500], name='wordvec_batch')
        music_variables,user_variables,prediction=CollabFilterring(user_batch, music_batch,word_vec_batch,'ConvAutoEnc')
        # rating_batch = tf.placeholder(tf.float32, [None,1], name='rating_batch')
        checkpoints_path='joint'
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoints_path)
        with tf.Session(graph=inference_graph) as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            music_latent=tf.get_default_graph().get_tensor_by_name('ConvAutoEnc/fc2/Relu:0')
            user_latent=tf.get_default_graph().get_tensor_by_name('inference/Sigmoid_1:0')

            
            for i,example in tqdm(enumerate(_examples)):
                arr=example.strip().split('\t')
                user_id=arr[0]
                track_id=arr[1]
                rating=arr[2]

                uesr_vec=user_input.get(user_id)[:]
                music_vec=music_input.get(track_id)
                word_vec=word_vec_input.get(track_id)

                if(music_vec is None):
                    cnt+=1
                    continue

                music_vec=music_vec[:] #这里一定要铺平，不然存不进去
                word_vec=word_vec[:]

                # music_vec=np.expand_dims(music_vec[],axis=0)
                # word_vec=np.expand_dims(word_vec,axis=0)
                uesr_vec=np.expand_dims(uesr_vec,axis=0)

                music_latent_vec, user_latent_vec = sess.run([music_latent, user_latent], feed_dict = {user_batch: uesr_vec, music_batch: music_vec, word_vec_batch:word_vec})
                # print(music_latent_vec)
                # print(user_latent_vec)
                if(user_id not in output_user.keys()):
                    output_user.create_dataset(user_id,data=user_latent_vec[0])
                if(track_id not in output_music.keys()):
                    output_music.create_dataset(track_id,data=music_latent_vec[0])

    
def training():
    user_batch = tf.placeholder(tf.float32, [None,2400], name='user_batch')
    music_batch = tf.placeholder(tf.float32, [None, 12,120,2], name='music_batch')
    wordvec_batch = tf.placeholder(tf.float32, [None, 500], name='wordvec_batch')
    rating_batch = tf.placeholder(tf.float32, [None,1], name='rating_batch')
    train_nn_mf(user_batch, music_batch,wordvec_batch, rating_batch)    


if __name__ == "__main__":
    # training()
    inference()

def printTime(remtime):
	hrs = int(remtime)//3600
	mins = int((remtime//60-hrs*60))
	secs = int(remtime-mins*60-hrs*3600)
	print("########  "+str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining  ########")


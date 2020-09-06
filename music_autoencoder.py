
import tensorflow as tf
import numpy as np
import h5py
from generate_music_tfrecord import read_tfRecord,load_train_lyrics,load_word_vec,parse,load_tfidf
import glob
from tqdm import tqdm 

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


def ConvAutoEncoder(x,vec, name):
    with tf.name_scope(name):
        # encoder
        c1 = conv2d(x, name='c1', kshape=[12, 120, 2, 16])
        p1 = maxpool2d(c1, name='p1')
        print(p1)
        do1 = dropout(p1, name='do1', keep_rate=0.8)
        c2 = conv2d(do1, name='c2', kshape=[6, 60, 16, 16])
        p2 = maxpool2d(c2, name='p2')
        print(p2)
        do2 = tf.reshape(p2, shape=[-1, 1440])
        print(do2)
        dot3=tf.concat([do2, vec], 1)
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
        print(output)
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))

        return output,cost

        # dc1 = deconv2d(do4, name='dc1', kshape=[5,5],n_outputs=25)

def training():
    batch_size = 4
    
    checkpoints_path='music_checkpoint/model'
    trainroad='data/music_train.tfrecords'
    with tf.Graph().as_default():
        iterator=read_tfRecord(trainroad)
        x = tf.placeholder(tf.float32, [None, 12,120,2], name='InputData')
        vec = tf.placeholder(tf.float32, [None, 500], name='InputData_vector')
        prediction,cost = ConvAutoEncoder(x, vec,'ConvAutoEnc')
        with tf.name_scope('opt'):
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        n_epochs=5
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for epoch in range(n_epochs):
                avg_cost = 0
                n_batches=100
                for i in range(n_batches):
                    image_batch,vector_batch=iterator.get_next()
                    image_batch,vector_batch=sess.run([image_batch,vector_batch])
                    image_batch=np.array(image_batch)
                    vector_batch=np.array(vector_batch)
                    fc1=tf.get_default_graph().get_tensor_by_name('ConvAutoEnc/Reshape:0')
                    _,c,fc1_value = sess.run([optimizer,cost,fc1], feed_dict={x: image_batch,vec:vector_batch})
                    # print(c)
                    # print(fc1_value)
                    # avg_cost += c / n_batches
                coord.request_stop()
                coord.join(threads)
                saver.save(sess, checkpoints_path)

def inference():
    example='/media/data/data/MillionSongSubset/data/A/A/A/TRAAABD128F429CF47.h5'
    dict_lyrics=load_train_lyrics()
    word_vec,id_word=load_word_vec()
    tfidf_vocab=load_tfidf('data/tfidf_train.txt')
    hdfFile = h5py.File(example, 'r')
    segments_pitches=hdfFile['analysis']['segments_pitches'][:]
    segments_timbre=hdfFile['analysis']['segments_timbre'][:]
    track_id=hdfFile['analysis']['songs']['track_id'][0].decode('UTF-8')
    print(track_id)
    lyrics=dict_lyrics[track_id]
    tfidfs=tfidf_vocab[track_id]
    vector=parse(lyrics,tfidfs,word_vec,id_word)
    while(segments_timbre.shape[0]<120):
        segments_timbre=np.concatenate((segments_timbre,segments_timbre),axis=0)
        segments_pitches=np.concatenate((segments_pitches,segments_pitches),axis=0)
    segments_pitches=segments_pitches[:120,:]
    segments_timbre=segments_timbre[:120,:]

    segments_pitches=segments_pitches.transpose(1,0)
    segments_timbre=segments_timbre.transpose(1,0)

    segments_pitches=np.expand_dims(segments_pitches,axis=2)
    segments_timbre=np.expand_dims(segments_timbre,axis=2)
    image=np.concatenate((segments_pitches,segments_timbre),axis=2)
    print(image.shape)
    inference_graph=tf.Graph()
    checkpoints_path='music_checkpoint'
    with inference_graph.as_default():
        x = tf.placeholder(tf.float32, [None, 12,120,2], name='InputData')
        vec = tf.placeholder(tf.float32, [None, 500], name='InputData_vector')
        prediction,cost = ConvAutoEncoder(x, vec,'ConvAutoEnc')
        saver = tf.train.Saver()
    output=h5py.File('data/music_vector.h5')
    with tf.Session(graph=inference_graph) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoints_path)   
        saver.restore(sess, ckpt.model_checkpoint_path)
        fc1=tf.get_default_graph().get_tensor_by_name('ConvAutoEnc/fc2/Relu:0')
        image=np.expand_dims(image,axis=0)
        vector=np.expand_dims(vector,axis=0)
        print(vector.shape)
        fc1_value = sess.run([fc1], feed_dict={x: image,vec:vector})
        fc1_value=np.array(fc1_value)
        print(fc1_value.shape)
        output.create_dataset(track_id,data=fc1_value[0])
    
        

def extract_data():
    dict_lyrics=load_train_lyrics()
    word_vec,id_word=load_word_vec()
    root_path='/media/data/data/MillionSongSubset/data/*/*/*/*.h5'
    tfidf_vocab=load_tfidf('data/tfidf_train.txt')
    h5files=glob.glob(root_path)
    output=h5py.File('data/music_vector.h5')
    music_input=h5py.File('data/music_input.h5')
    music_wv=h5py.File('data/music_wv.h5')

    inference_graph=tf.Graph()
    checkpoints_path='music_checkpoint'
    with inference_graph.as_default():
        x = tf.placeholder(tf.float32, [None, 12,120,2], name='InputData')
        vec = tf.placeholder(tf.float32, [None, 500], name='InputData_vector')
        prediction,cost = ConvAutoEncoder(x, vec,'ConvAutoEnc')
        saver = tf.train.Saver()
    cnt=0
    with tf.Session(graph=inference_graph) as sess:

        ckpt = tf.train.get_checkpoint_state(checkpoints_path)   
        saver.restore(sess, ckpt.model_checkpoint_path)
        fc1=tf.get_default_graph().get_tensor_by_name('ConvAutoEnc/fc2/Relu:0')
        for h5file in tqdm(h5files):
            hdfFile = h5py.File(h5file, 'r')
            segments_pitches=hdfFile['analysis']['segments_pitches'][:]
            segments_timbre=hdfFile['analysis']['segments_timbre'][:]
            track_id=hdfFile['analysis']['songs']['track_id'][0].decode('UTF-8')
            song_id=hdfFile['metadata']['songs']['song_id'][0].decode('UTF-8')
            if(track_id in dict_lyrics):
                lyrics=dict_lyrics[track_id]
                tfidfs=tfidf_vocab[track_id]
                vector=parse(lyrics,tfidfs,word_vec,id_word)
                
            else:
                continue

            while(segments_timbre.shape[0]<120):
                segments_timbre=np.concatenate((segments_timbre,segments_timbre),axis=0)
                segments_pitches=np.concatenate((segments_pitches,segments_pitches),axis=0)
            segments_pitches=segments_pitches[:120,:]
            segments_timbre=segments_timbre[:120,:]

            segments_pitches=segments_pitches.transpose(1,0)
            segments_timbre=segments_timbre.transpose(1,0)

            segments_pitches=np.expand_dims(segments_pitches,axis=2)
            segments_timbre=np.expand_dims(segments_timbre,axis=2)
            image=np.concatenate((segments_pitches,segments_timbre),axis=2)
            # print(image.shape)
            image=np.expand_dims(image,axis=0)
            vector=np.expand_dims(vector,axis=0)
            # print(vector.shape)
            music_input.create_dataset(song_id,data=image)
            music_wv.create_dataset(song_id,data=vector)

            fc1_value = sess.run([fc1], feed_dict={x: image,vec:vector})
            fc1_value=np.array(fc1_value)
            # print(fc1_value.shape)
            output.create_dataset(track_id,data=fc1_value[0])
            cnt+=1
        print(cnt)





 
if __name__ == '__main__':
    
    # training() 
    # inference() 
    extract_data()  # extract music latent vectors
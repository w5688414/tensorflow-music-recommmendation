import fasttext
import pandas as pd
import numpy as np
# Skipgram model
# model = fasttext.skipgram('data.txt', 'model')
# print(model.words) # list of words in dictionary
 
# print(model['machine']) # get the vector of the word 'machine'
def train():
    model = fasttext.train_unsupervised("data.txt", model='cbow',thread=30,minn=3,maxn=6,loss='softmax', lr=0.05, dim=500, ws=5, epoch=1)
    model.save_model("model_file.bin")

def test():
    model = fasttext.load_model("model_file.bin")
    # print(model.words)   # list of words in dictionary
    print(model['the'].shape) # get the vector of the word 'king'


def get_word_index():
    list_word_index=[]
    with open('data/words.txt','r') as f:
        for item in f.readlines():
            print(item)
            key,index=item.strip().split('\t')
            list_word_index.append([key,index])
    return list_word_index

def output_to_csv(word_index,list_vec):
    
    list_res=[]
    print(type(list_vec[0]))
    for i in range(len(word_index)):
        list_res.append([word_index[i][0],word_index[i][1]])
    
    column_name = ['word', 'index']
    list_vec=np.array(list_vec)
    np.save('data/vec.npy',list_vec)
    csv_name='data/word_to_vec.csv'
    xml_df = pd.DataFrame(list_res, columns=column_name)
    xml_df.to_csv(csv_name, index=None) 

def process_word_vec():
    word_index=get_word_index()
    model = fasttext.load_model("model_file.bin")
    list_vec=[]
    for word in word_index:
        vec=model[word[0]]
        list_vec.append(vec)
    output_to_csv(word_index,list_vec)





if __name__ == "__main__":
    # test()
    process_word_vec()
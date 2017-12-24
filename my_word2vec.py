# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:24:50 2017

@author: LIKS
"""
import tensorflow as tf
from six.moves import urllib
from tempfile import gettempdir
import os
import zipfile
import collections
import numpy as np
import random
import math

url="http://mattmahoney.net/dc/"

#step1:down load and read the needed document 



def maybe_download(filename,expected_bytes):
    def cbk(a,b,c):
      per=100*a*b/c
      if per>100:
          per=100
      print(str(per)+'% finished')
    local_filename=os.path.join(gettempdir(),filename)
    if not os.path.exists(local_filename):
        local_filename,_=urllib.request.urlretrieve(url+filename,local_filename,cbk)
    statinfo=os.stat(local_filename)
    if statinfo.st_size==expected_bytes:
        print('Found and verfied', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify'+local_filename+'.Can you get to it with a browser')
    return local_filename

filename=maybe_download('text8.zip',31344016)
        
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary=read_data(filename)
print("Data size",len(vocabulary))

#step2:build the dictionary and replace the rare words with UNK token. digitalize the vocabulary
vocabulary_size=50000



def build_dataset(words,n_words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
     #将文本变为数字，即数字化   
    data=list()
    unk_count=0
    for word in words:
        #用这个更简洁：dictionary.get（word,defaut=0）,存在返回值不存在返回0
        index=dictionary.get(word,0)
        if index==0:
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count,dictionary, reverse_dictionary=build_dataset(vocabulary, vocabulary_size)

del vocabulary
print('Most Common Words (+UNK):',count[:5])
print('Sample data', data[:10],[reverse_dictionary[i] for i in data[:10]])
    
#step3:function to generate a training batch for the skip gram model
data_index=0

def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    label=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1
    buffer=collections.deque(maxlen=span)
#   below code is also ok 
#   for _ in range(span):
#        buffer.append(data[data_index])
#        data_index=(data_index+1)%len(data)
    if data_index+span>len(data):
        data_index=0
    buffer.extend(data[data_index:data_index+span])
    data_index+=span
    #OUTER RECYCLE
    for i in range(batch_size//num_skips):
        context_words=[w for w in range(span) if w != skip_window]
        #每个batch产生随机顺序的训练数据
        words_to_use=random.sample(context_words,num_skips)
        #INNER RECYCLE, USE ENUMERATE
        for j,context_word in enumerate(words_to_use):
            batch[i*num_skips+j]=buffer[skip_window]
            label[i*num_skips+j,0]=buffer[context_word]
       #judge if out ot range before moving to next index
        if data_index==len(data):
            buffer[:]=data[:span]
            data_index=span
        else:    
            buffer.append(data[data_index])
            data_index+=1
   #backtrack a little bit to avoid skipping words in the end of a batch?????
    data_index=(data_index+len(data)-span)%len(data)
    return batch,label

#test some samples
batch,label=generate_batch(8,2,1)
for i in range(8):
    print(reverse_dictionary[batch[i]],batch[i],
          '-->',reverse_dictionary[label[i,0]],label[i,0])
    
#built and train a skip gram model
batch_size=128
embedding_size=128
skip_window=1
num_skips=2
num_sampled=64 #negative samples

valid_size=16 # to evaluate similarity on
valid_window=100
valid_examples=np.random.choice(valid_window,valid_size,replace=False)

graph=tf.Graph()

with graph.as_default():
    #input data
    train_inputs=tf.placeholder(tf.int32,shape=[batch_size])#不是embedding_size吗？（不是，因为数据还需要进一步处理）
    train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,tf.int32)
    #pinned to CPU because of missing GPU implementation
    with  tf.device('/cpu:0'):
        #look up embeddings for input
        embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
        embed=tf.nn.embedding_lookup(embeddings,train_inputs)
        
        #construct variables for NCE LOSS
        #math.sqrt和tf.sqrt有区别吗？
        nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
        nce_biases=tf.Variable(tf.zeros([vocabulary_size]))
        
    loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                           biases=nce_biases,
                                           labels=train_labels,
                                           inputs=embed,
                                           num_sampled=num_sampled,
                                           num_classes=vocabulary_size))
    optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings=embeddings/norm
    #normalized_embeddings:50000*128,16*1: return:16*128?
    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity=tf.matmul(valid_embeddings,normalized_embeddings, transpose_b=True)
    
    init=tf.global_variables_initializer()

# start to train
num_steps=100001
with tf.Session(graph=graph) as sess:
    init.run()
    print('Initialized!')
    
    average_loss=0
    for step in range(1,num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size,num_skips,skip_window)
        _,loss_val=sess.run([optimizer,loss],feed_dict={train_inputs:batch_inputs,train_labels:batch_labels})
        average_loss+=loss_val
        
        
        if step%2000==0:
            average_loss/=2000
            print('Average loss at step ',step,' : ',average_loss)
            average_loss=0
            
        if step%10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=reverse_dictionary[valid_examples[i]]
                top_k=8
                nearest=(-sim[i,:]).argsort()[1:top_k+1]#数组[1:8]的区间[1,8)，索引为0的是其自身
                log_str='Nearest to %s:'%valid_word
                for k in range(top_k):
                    close_word=reverse_dictionary[nearest[k]]
                    log_str='%s %s,'%(log_str, close_word)
                print(log_str)
    final_embeddings=normalized_embeddings.eval()

def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert low_dim_embs.shape[0]>=len(labels),'more labels than embeddings'
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_only=100
low_dim_embs=tsne.fit_transform(final_embeddings[:plot_only,:])
labels=[reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs,labels)
                    
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    





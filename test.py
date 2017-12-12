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
    awo
    
    
    
    
    
    
    
    
    
    
    
    





import pandas as pd
import numpy as np
import sys
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import multiprocessing

import yaml
import keras
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

cpu_count = multiprocessing.cpu_count()
sys.setrecursionlimit(1000000)

vocab_dim = 100
n_iterations = 10
n_exposures = 10 # 词典统计所有频数超过这个值的词语
window_size = 7
n_epoch = 4
input_length = 100
maxlen = 100
batch_size=32

#读取数据集和标签生成
def read_data():
    #导入训练数据，分为正负和中性三种
    neg=pd.read_csv('./data/neg.csv',header=None,index_col=None)
    pos=pd.read_csv('./data/pos.csv',header=None,index_col=None,error_bad_lines=False)
    neu=pd.read_csv('./data/neu.csv', header=None, index_col=None)

    #将三类数据整合在一起
    combined = np.concatenate((pos[0], neu[0], neg[0]))
    #注意这边要取出值来整合，不加【0】就是一个列表地进行合并[[],[],[],..]，（xxxx,1）;加上就是（xxxx,）,[...]

    #pos设为1，neu设为0，neg设为-1
    label=np.concatenate((np.ones(len(pos),dtype=int),np.zeros(len(neu),dtype=int),-1*np.ones(len(neg),dtype=int)))

    return combined,label

#对句子经行分词，并去掉换行符
def tokenizer(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

#创建词语字典，包含词到索引的映射、词到向量的映射、每个句子所对应的词语索引
def create_dictionaries(model=None,
                        text=None):
    if (text is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        #  索引为0的代表频数小于设定值的词，所以这里k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过设定频数的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过设定频数的词语的词向量, (word->model(word))

        def parse_dataset(text):
            data=[]
            for sentence in text:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) #频数小于设定值的词，索引为0
                data.append(new_txt)
            return data  # 词变为索引

        text=parse_dataset(text)
        text= sequence.pad_sequences(text, maxlen=maxlen)
        #每个句子所含词语对应的索引，所以句子中含有频数小于设定频数的词语，索引为0
        return w2indx, w2vec,text
    else:
        print ('模型或数据空')

def word2vec_train(text):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(text)
    model.train(text,total_examples=model.corpus_count,epochs=model.iter)
    model.save('../model/Word2vec_model.pkl')
    index_dict, word_vectors,text = create_dictionaries(model=model,text=text)
    return index_dict, word_vectors,text

def get_data(index_dict, word_vectors, text, label):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于设定值的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x = text
    y = keras.utils.to_categorical(label, num_classes=3)
    return n_symbols, embedding_weights,x,y


##定义网络结构
def init_lstm(n_symbols, embedding_weights):
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(output_dim=50, activation='tanh', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 全连接层,输出维度=3，经softmax后是每个类别的概率
    model.add(Activation('softmax'))

    # 设置损失和优化器
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm(model, x, y):
    model.fit(x, y, batch_size=batch_size, epochs=n_epoch, verbose=1)
    return model

def save_model(model)
    yaml_string = model.to_yaml()
    with open('../model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../model/lstm.h5')


text,label=read_data()
text = tokenizer(text)
print('开始训练word2vec模型...')
index_dict, word_vectors,text=word2vec_train(text)
print('结束训练word2vec模型')

n_symbols, embedding_weights, x, y= get_data(index_dict, word_vectors, text, label)
model=init_lstm(n_symbols, embedding_weights)
print("开始训练LSTM...")
model = train_lstm(model,x, y)
print("结束训练LSTM")
save_model(model)
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import jieba
import jieba.analyse as analyse
import yaml
from keras.models import model_from_yaml

import sys
sys.setrecursionlimit(1000000)
#忽略所有警告
import warnings
warnings.filterwarnings('ignore')

maxlen = 100

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


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('./model/Word2vec_model.pkl')
    _,_,text=create_dictionaries(model,words)
    return text


def lstm_predict(model,string):
    data=input_transform(string)
    data.reshape(1,-1)
    #print(data)
    result=model.predict(data)

    #返回string是积极的的概率
    return result[0]



with open('./model/lstm.yml', 'r') as f:
    yaml_string = yaml.load(f)
model = model_from_yaml(yaml_string)
model.load_weights('./model/lstm.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

while 1:
    print('请输入文字(回车退出)：')
    string=input()
    if string=='' :
        break
    else:
        print('这句话是好评的概率为：',lstm_predict(string))

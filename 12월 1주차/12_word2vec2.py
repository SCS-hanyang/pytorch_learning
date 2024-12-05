import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/"
#                           "main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename=r"data\ted_en-20160408.xml")

filename = r"data\ted_en-20160408.xml"

targetXML = open(filename, 'r', encoding='UTF8')
target_text = etree.parse(targetXML)
'''
XML의 계층적 구조

    각 tag들이 노드를 이루고, tag의 포함 관계에 따라 부모-자식간의 관계가 결정됨
    .tag를 통해서 tag에 접근할 수 있고, .text를 통해서 안에 있는 content를 접근할 수 있다
'''

parse_text = '\n'.join(target_text.xpath('//content//text()'))  # 각 요소들의 결합 사이에 \n 추가한다는 의미
                                                                # 예시 : print('\n'.join(('aaa', 'bbb')))


content_text = re.sub(r'\([^)]*\)', '', parse_text)

sent_text = sent_tokenize(content_text)                         # 문장 텍스트를 각 문장 단위로 분리

normalized_text = []

for sent in sent_text:
    sent = re.sub(r'[^a-z0-9]+', ' ', sent.lower())    # string.lower() 모든 문자를 소문자로
    normalized_text.append(sent)

text = [word_tokenize(sentence) for sentence in normalized_text]

print(text[:10])

#model = Word2Vec(sentences=vocab, vector_size=100, window=5, min_count=5, workers=4, sg=0)
'''
vector_size : 임베딩 된 벡터의 차원
min_count : 단어 최소 빈도 수
workers : 학습을 위한 프로세서 수
sg : 0은 CBOW, 1은 Skip-gram

'''
'''
model_result = model.wv.most_similar("man")
print(model_result)

model.wv.save_word2vec_format(r'model\eng_w2v')            # 모델 저장

loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드
'''
'''
import gensim
import urllib.request

# 구글의 사전 훈련된 Word2Vec 모델을 로드.
urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", \
                           filename="GoogleNews-vectors-negative300.bin.gz")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
'''


def text2vocab(text, min_count):
    word_counter = Counter()
    for sentence in text:
        word_counter.update(sentence)

    idx2word = {word for word, count in word_counter.items() if count >= min_count}
    return idx2word


idx2word = text2vocab(text, 5)
vocab_size = len(idx2word)

for idx, sentence in enumerate(text):
    for word in sentence:
        if word not in idx2word:
            sentence.remove(word)

class myWord2Vec(nn.Module):
    def __init__(self, vector_size, window):
        super(myWord2Vec, self).__init__()
        self.window = window
        self.layer1 = nn.Linear(in_features=self.vocab_size, out_features=vector_size, bias=False)
        self.layer2 = nn.Linear(in_features=vector_size, out_features=self.vocab_size, bias=True)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.mean(x, dim=0)
        x = self.layer2(x)
        return x
    
model = myWord2Vec(100, 5)
nb_epoch = 15

for epoch in range(nb_epoch):
    for sentence in text:
        for idx, central_word in enumerate(sentence):
            if idx < model.window:
                x = torch.tensor(sentence[:idx]+sentence[idx+1:idx+model.window+1], dtype=torch.float32)
            elif idx+model.window+1 > len(sentence):
                x = torch.tensor(sentence[idx-model.window:idx]+sentence[idx+1:], dtype=torch.float32)
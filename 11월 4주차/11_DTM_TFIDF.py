import torch

'''

Document-Term Matrix, DTM(문서 단어 행렬)

    :다수의 문서에 등장하는 각 단어들의 빈도를 행렬로, 간단하고 구현 쉬움
    
    한계점
        1) Sparse representation(희소 표현)
            : 각 문서의 벡터의 차원은 전체 단어 표현의 크기 -> 과도하게 큰 벡터 차원을 가진다 -> 계산 복잡
            : 문서끼리 공유하는 단어의 개수가 적을 수 있어서, 대부분의 값들이 0이 될 확률도 높다
            해결책 : 전처리 과정을 통해서 불용어나 빈도수가 낮은 단어 제거, 어간 및 표제어 추출
        
        2) 단순 빈도 수 기반 접근
            : 공유되는 단어의 빈도가 많다고 문서끼리 유사한 문서라고 판단 X(the 같은 단어가 많이 공유된다고 유사한 문서가 아니기때문)
            
TF-IDF(Term Frequency-Inverse Document Frequency, 단어 빈도-역 문서 빈도) 
    
    : DTM 내의 각 단어들마다 중요한 정도로 가중치 부여
    : 단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함) 사용
    : TF-IDF = TF * IDF

    1) tf(d,t)  : 특정 문서 d에서의 특정 단어 t의 빈도
    2) df(t)    : 특정 단어 t가 등장한 문서 수
    3) idf(t)   : df(t)에 반비례하는 수
                : log(n / (1+df(t))
                : n은 총 문서의 수
                
    : tf-idf는 모든 문서에서 자주 등장하는 단어는 중요도가 낮다고 판단, 특정 문서에서만 자주 등장하면 중요도가 높다고 판단
    
Euclidean distance

Jaccard similarity
'''

import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

def tf(t,d):
    return d.count(t)    # str.count(obj) : str안에서 obj 개수 찾기

def idf(t):
    N = len(docs)
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (1 + df))

def tf_idf(t, d):
    return tf(t,d) * idf(t)

result = []

for doc in docs:
    result.append([0]*len(vocab))
    for idx,word in enumerate(vocab):
        result[-1][idx] += tf(word, doc)

tf_ = pd.DataFrame(data=result, columns = vocab)

results = []

for doc in docs:
    results.append([0]*len(vocab))
    for idx,word in enumerate(vocab):
        results[-1][idx] += tf_idf(word, doc)

tfidf_ = pd.DataFrame(data=results, columns = vocab)

print(tf_)
print(tfidf_)

'''
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()
vector.fit_transform(corpus).toarray())         DTM 함수
'''

'''
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())       TF-IDF 함수

'''
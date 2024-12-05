import torch

'''
Word Embedding

    단어를 벡터로 표현
    방법론으로는 LSA, Word2Vec, FastText, Glove, nn.embedding 등이 있다
    
Sparse Representation

    원 핫 인코딩의 경우, 대부분의 값이 0이 된다. 이를 sparse representation이라고 하고, 원 핫 벡터를 sparse vector이다
    원 핫 인코딩의 경우 공간 낭비가 심하며, 단어간 유사성을 나타낼 수 없다
    
Dense Representation

    벡터의 차원을 단어 집합의 크기로 상정 X
    단어를 dense vector로 표현하는 것이 word embedding
    
Word2Vec

    distributed representation : 단어의 의미를 다차원 공간에 벡터화
    이는 distributed hypothesis에서 만들어졌는데, "비슷한 위치에서 등장하는 단어는 비슷한 의미이다"
    CBOW와 Skip-Gram으로 학습됨
    
CBOW(continuous bag of words)
    
    주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측
    중심 단어를 예측하기 위해 앞, 뒤로 몇 개의 단어를 볼지를 결정하면, 이 범위를 window라고 함(window : n -> 참고 단어 : 2n)
    윈도우가 결정된 이후 윈도우를 계속 움직이면서 예측 : sliding window
    
    input layer : 윈도우 범위 안에 있는 단어들의 one-hot vector
    output layer : 예측하고자 하는 중간 단어의 one-hot vector
    input, output layer 안에 하나의 hidden layer만이 존재
    
    Projection layer 의 크기를 M이라고 하면, 그 값은 CBOW를 수행하고 나서 얻는 각 단어의 임베딩 벡터의 차원
    input -> projection, projection -> output 로의 weight matrix W, W'을 학습하면서 중심 단어의 예측을 더 잘 되게 학습
    
    input -> projection
        
        input vector가 원 핫 벡터이기 때문에 x.matmul(W)을 하면 W의 i번째 행렬이 추출된다. 이를 lookup이라고 한다.
        이렇게 2n개의 주변 단어들을 통해 lookup해온 벡터들의 평균을 구하여 projection layer의 값으로 설정한다
        
    projection -> output
        
        이 벡터를 가지고 중간 단어의 원 핫 벡터 예측을 위해 W'를 학습한다
        W'와 곱해진 이후 softmax 함수를 지나게 되는데, 이 결과값을 score vector라고 한다 -> 그 후 cross entropy로 중간 단어 예측
        
    최종적인 단어의 dense vector는 단어의 원 핫 벡터가 xi일 경우 matmul(xi, W)로 결정됨
    
Skip-Gram

    중간에 있는 단어들을 가지고, 주변 단어들을 예측
    
    CBOW와 반대로
    
    input layer : 중간 단어의 원 핫 벡터
    output layer : 주변 단어의 원 핫 벡터
    
    CBOW보다 성능 좋음
    
Negative sampling
    
    일반적으로 Word2Vec을 사용한다고 하면, Negative Sampling이란 방법까지 사용하는 것(SGNS : Skip Gram with negative sampling)
    
    Word2Vec에서 softmax 함수를 계산할 경우, 단어 벡터의 전체 집합애 대해 softmax 값을 구해야 한다. 이 때, 어휘 집합의 크기가 크면 계산 속도가 느려진다
    그래서, 단어 전체 집합에 대한 softmax 함수를 계산하는 것이 아닌, 중심단어 혹은 주변 단어와 무관한 negative sample들과
    중심 단어와 주변 단어들인 positive sample들로 손실 함수를 계산하는 것이 negative sampling이다
    
    손실 함수 J = log sigmoid(positive sample) + sigma(log sigmoid(-negative sample))을 통해 positive sample들과 유사도를 최대화하고
    negative sample들과 유사도를 최소화한다
    
    
'''
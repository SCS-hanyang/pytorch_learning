import torch
import torch.nn as nn
'''
embedding vector
    : lookup table에 one-hot vector를 곱해져서 dense vector가 나오는 것을 embedding vector라고 한다.
      학습 시, lookup table을 학습시킨 다음, 구하고자 하는 단어의 idx로 embedding vector 구할 수 있다.

'''

# nn.embedding 없이 구현

train_data = 'you need to know how to code'

word_set = set(train_data.split(' '))

vocab = {word : i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

sample = 'you need to run'.split()
idxes = []

for word in sample:
    try:
        idxes.append(vocab[word])

    except KeyError:
        idxes.append(vocab['<unk>'])

idxes = torch.LongTensor(idxes)

lookup_result = embedding_table[idxes, :]           # 특정 idx 리스트를 통해 요소 가져오기
print(lookup_result)


# nn.embedding 으로 구현

train_data = 'you need to know how to code'

word_set = set(train_data.split(' '))

vocab = {word : i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3, padding_idx=1)       # pad idx : lookup table에서 0벡터로
                                                                                                # 고정된 행 벡터. 그 벡터는 input의 길이를
                                                                                                # 맞추기 위한 padding에 할당됨
print(embedding_layer.weight)       # 모든 벡터에 대해 requires_True 이지만, 내부 계산으로 gradient가
                                    # 계산 되지만 업데이트 되지는 않는다


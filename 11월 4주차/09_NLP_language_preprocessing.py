<<<<<<< HEAD
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
import urllib.request
import pandas as pd
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np


en_text = "A Dog Run back corner near spare bedrooms"
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

#kor_tokenizer = Okt()

en_token = word_tokenize(en_text)
#kor_token = kor_tokenizer.morphs(kor_text)


#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')

print(f"data 형식 \n {data[:1]}")

sample_data = data[:100]

sample_data.loc[:,'document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)    # str.replace = 문자열 변환, regex=True : 조건이 정규표현식이라는 것을 나타냄
                                                                                                           # dataframe.loc[행, 열]을 통해, 원본 dataframe을 수정한다는 사실 전달
                                                                                                           # 아니면 pandas가 혼란해함
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = Okt()

token = []
for data in sample_data['document']:
    data = tokenizer.morphs(data)
    temp = [c for c in data if c not in stopwords]
    token.append(temp)

vocab = FreqDist(np.hstack(token))          # FreqDist : 단어 및 토큰의 반복 횟수 게산 -> 사전처럼 동작하는 자료형
                                            # np.hstack : 리스트를 수평으로 평탄화

vocab_size = 500

vocab = vocab.most_common(vocab_size)

word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []

for line in token:
    temp = []
    for word in line:
        try:
            temp.append(word_to_index[word])
        except KeyError:
            temp.append(word_to_index['unk'])

    encoded.append(temp)

max_len = max(len(l) for l in encoded)

for value in encoded:
    if len(value) < max_len:
        value += [word_to_index['pad']] * (max_len - len(value))

=======
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
import urllib.request
import pandas as pd
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np


en_text = "A Dog Run back corner near spare bedrooms"
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

#kor_tokenizer = Okt()

en_token = word_tokenize(en_text)
#kor_token = kor_tokenizer.morphs(kor_text)


#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')

print(f"data 형식 \n {data[:1]}")

sample_data = data[:100]

sample_data.loc[:,'document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)    # str.replace = 문자열 변환, regex=True : 조건이 정규표현식이라는 것을 나타냄
                                                                                                           # dataframe.loc[행, 열]을 통해, 원본 dataframe을 수정한다는 사실 전달
                                                                                                           # 아니면 pandas가 혼란해함
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = Okt()

token = []
for data in sample_data['document']:
    data = tokenizer.morphs(data)
    temp = [c for c in data if c not in stopwords]
    token.append(temp)

vocab = FreqDist(np.hstack(token))          # FreqDist : 단어 및 토큰의 반복 횟수 게산 -> 사전처럼 동작하는 자료형
                                            # np.hstack : 리스트를 수평으로 평탄화

vocab_size = 500

vocab = vocab.most_common(vocab_size)

word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []

for line in token:
    temp = []
    for word in line:
        try:
            temp.append(word_to_index[word])
        except KeyError:
            temp.append(word_to_index['unk'])

    encoded.append(temp)

max_len = max(len(l) for l in encoded)

for value in encoded:
    if len(value) < max_len:
        value += [word_to_index['pad']] * (max_len - len(value))

>>>>>>> 95b60bdff058f123cb273be7163370ffb90b989b

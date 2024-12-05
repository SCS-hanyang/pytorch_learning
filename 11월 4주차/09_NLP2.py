from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

'''
코퍼스에서 용도 맞게 토큰 분류 -> tokenization

토큰화 후 텍스트 데이터의 용도에 맞게 cleaning & normalization 진행
    cleaning : 갖고 있는 코퍼스로부터 노이즈 데이터를 제거한다
    normalization : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만든다
    
노이즈 데이터의 종류
    1. 등장 빈도가 적은 단어
    2. 길이가 짧은 단어(주로 영어에서, 한글은 적용이 애매함 -> 이는 한글이 가진 함축적 특성 때문(한자어여서))
    
Regular Expression
    노이즈 데이터의 특성을 파악하였다면 regular expression 을 통해 한번에 제거 가능
    
불용어(Stopwords)
    갖고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업
    큰 의미가 없다는 뜻은 자주 등장하지만 분석에 도움이 안되는 단어(i, me, my 등)
'''

stop_word_list = stopwords.words("english")
print(f"number of stopwords : {len(stop_word_list)}")
print(f'불용어 10개 출력', stop_word_list[:10])

# 불용어 제거하기

example = 'when you have faults, do not fear to abandon them'
stop_word_list = set(stop_word_list)

word_token = word_tokenize(example)

list=[]

for token in word_token:
    if token not in stop_word_list:
        list.append(token)

print(f"sentence before stopwords removal : {word_token}")
print(f"sentence after stopwords removal : {list}")
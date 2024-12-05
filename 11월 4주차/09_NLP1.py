<<<<<<< HEAD
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tag import pos_tag

from konlpy.tag import Okt
from konlpy.tag import Kkma
import jpype

# corpus :  텍스트 데이터의 집합을 의미하며, 언어 연구와 처리를 위한 기초 자료로 사용,
# 특정 언어, 주제, 도메인, 또는 사용 목적에 따라 수집된 텍스트 데이터로 구성

sentence = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(sentence)

print("단어 토큰 : ", tokenized_sentence)
print("품사 태깅 : ", pos_tag(tokenized_sentence))

okt = Okt()
kkma = Kkma()

print("OKT 형태소 분석", okt.morphs("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
print("OKT 품사 태깅", okt.pos("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
print("OKT 명사 추출", okt.nouns("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))

print("KKMA 형태소 분석", kkma.morphs("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
print("KKMA 품사 태깅", kkma.pos("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
=======
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tag import pos_tag

from konlpy.tag import Okt
from konlpy.tag import Kkma
import jpype

# corpus :  텍스트 데이터의 집합을 의미하며, 언어 연구와 처리를 위한 기초 자료로 사용,
# 특정 언어, 주제, 도메인, 또는 사용 목적에 따라 수집된 텍스트 데이터로 구성

sentence = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(sentence)

print("단어 토큰 : ", tokenized_sentence)
print("품사 태깅 : ", pos_tag(tokenized_sentence))

okt = Okt()
kkma = Kkma()

print("OKT 형태소 분석", okt.morphs("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
print("OKT 품사 태깅", okt.pos("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
print("OKT 명사 추출", okt.nouns("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))

print("KKMA 형태소 분석", kkma.morphs("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
print("KKMA 품사 태깅", kkma.pos("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
>>>>>>> 95b60bdff058f123cb273be7163370ffb90b989b
print("KKMA 명사 추출", kkma.nouns("젠레스 존 제로라는 게임을 아시나요? 정말 갓겜입니다!"))
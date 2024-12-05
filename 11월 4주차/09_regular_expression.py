import re       # 정규 표현식

r'''
정규 표현식 문법

.           : 한 개의 임의의 문자
?           : 앞의 문자가 문자가 0 or 1 존재
*           : 앞의 문자가 0개 이상
+           : 앞의 문자가 1개 이상
^           : 뒤의 문자열로 문자열 시작
$           : 앞의 문자열로 문자열 끝
{num}       : num만큼 반복
{num1, num2}: num1~num2 만큼 반복, ?,*,+를 이것으로 대체 가능
{num,}      : num 이상 만큼 반복
[str]       : 대괄호 안의 문자들 중 한 개의 문자와 매칭, [a-z]와 같이 범위 지정 가능, [a-zA-Z] 전 범위 알파벳과 매칭
[^str]      : 해당 문자를 제외한 문자를 매치
l           : AlB같이 쓰이며 A 또는 B

\\\         : \ 문자 자체 의미
\\d         : 모든 숫자 의미 == [0-9]
\\D         : 숫제 제외 모든 문자 == [^0-9]
\\s         : 공백
\\S         : 공백 제외 모든 문자
\\w         : 문자 또는 숫자 == [0-9a-zA-Z]
\\W         : 문자 또는 숫자가 아닌 모든 문자 == [^0-9a-zA-Z]
'''

'''

정규 표현식 모듈 함수

re.compile()    : 정규 표현식 컴파일
re.search()     : 문자열 전체에 대해 정규표현식과 매치되는지 검색
re.match()      : 문자열 처음이 매치되는지
re.split()      : 정규 표현식 기준으로 문자열 분리 -> 리스트
re.findall()    : 문자열에서 매치되는 모든 문자열 찾아 리스트, 없으면 빈 리스트
re.finditer()   : 매치되는 모든 문자열 이터레이터 객체로 리턴
re.sub()        : 일치하는 부분 다른 문자열로


'''

r1 = re.compile('a.c')
str1 = r1.search('akcald')
print(str1)

r2 = re.compile('ab?c')
str21 = r2.search('ac')
str22 = r2.search('abc')
print(str21, str22)

r3 = re.compile('ab{2}c')
str31 = r3.search('abc')
str32 = r3.search('abbc')
print(str31, str32)


# 공백 기준 분리
text = "사과 딸기 수박 메론 바나나"
split = re.split(" ", text)
print(split)

text2 = "사과+딸기+수박+메론+바나나"

split2 = re.split("\\+", text2)         # +가 의미 있는 기호이기 때문에 + 자체를 나타내려면 \\+로 써야함
print(split2)

text3 = "사과a딸기a수박a메론a바나나"

split3 = re.split("a", text3)

print(split3)

text = "Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."

preprocessed_text = re.sub('[^a-zA-Z]', ' ', text)      # 알파벳 아닌거 전부 공백으로 치환
print(preprocessed_text)

text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""

split = re.split('\\s+', text)

print(split)
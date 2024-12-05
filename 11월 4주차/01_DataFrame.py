import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)

X = df['메일 본문']
y = df['스팸 메일 유무']

print('X 데이터 :',X)
print('y 데이터 :',y.to_list())
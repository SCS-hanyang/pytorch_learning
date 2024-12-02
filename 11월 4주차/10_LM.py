import torch

'''
n-gram
    
    : 언어 모델의 확률 계산은, 이전 단어를 관측한 다음 다음 단어를 예측하는 것이다. 하지만 이 경우, sequence의 길이가 길어질수록 학습한
      dataset에 그 sequence가 모두 들어가 있을 확률이 적다. 이를 sparsity problem이라고 한다. 그렇기 때문에, 예측 확률을 높이고자 
      현재 위치에서 앞에 있는 n개의 단어들로 다음 단어를 예측하는 것이 n-gram이다. 실험 결과 n은 5 이상이면 안된다
      
    trade-off : 하지만 당연하게도, n이 작을수록 정확도는 떨어진다.
    
단어의 표현 방법

    1. 국소 표현(Local / Discrete Representation) : 해당 단어만 보고, 특정값을 매핑하여 단어를 표현
    2. 분산 표현(Distributed / Continuous Representation) : 그 단어를 표현하고자, 주변 참조
    
Bag of Words
    
    : 단어의 순서를 고려하지 않고, 출현 빈도 수에 집중
        1. 각 단어에 정수 인덱스 부여
        2. 각 정수 인덱스에 출현 빈도가 부여된 벡터 생성
    
'''
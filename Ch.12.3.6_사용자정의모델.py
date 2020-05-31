# %%
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import copy
import random

# %%
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers,  n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation = "elu", kernel_initalizer = "he_normal") for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z
# %%
class ResidualRegressor(keras.Model):

    # 잔차회귀 (ResidualRegressor) 신경망 모델의 필수 매개변수들 정의
    ## 보통 신경망 모델에 사용될 각 층들을 정의한다.
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation = "elu", kernel_initializer = "he_normal") # 뉴런 개수가 30개인 완전연결 입력층을 인스턴스 속성으로 정의
        self.block1 = ResidualBlock(2, 30) # 첫번째 잔차블록층을 인스턴스 속성으로 정의 : n_layers = 2, n_neurons = 30
        self.block2 = ResidualBlock(2, 30) # 두번째 잔차블록층을 인스턴스 속성으로 정의 :
        self.out = keras.layers.Dense(output_dim) # 뉴런 개수가 output_dim개인 완전연결 출력층을 인스턴스 속성으로 정의

    # 잔차회귀 신경망 모델의 순전파 프로시져를 정의
    ## 입력값이 입력층 - 은닉층 - 출력층을 통과하는 작동 방식을 정의한다.
    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

# %%

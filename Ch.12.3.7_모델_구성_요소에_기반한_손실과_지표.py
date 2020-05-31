# %%
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import copy
import random

# %%
class ReconstructionRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation = "selu", kerel_initialization = "lecun_normal") for _ in range(5)] # Dense() 객체를 5회 호출하여 리스트 형태로 담은뒤 self.hidden 인스턴스 속성으로 정의한다.
        self.out = keras.layers.Dense(output_dim) # output_dim을  Dense() 객체의 n_neurons에 해당하는 매개변수로 전달하여 self.out 인스턴스 속성으로 정의한다.

    # 보조 손실 (재구성 손실) 정의
    ## 모델 구성 요소 중 '완전 연결층'을 활용해 재구성 손실을 계산할 수 있다. 재구성 손실을 계산하기 위해 구축된 층을 재구성 층이라고 한다.
    ## 재구성 손실은 어느 한 완전 연결층이 반환하는 잠재행렬 (represented matrix) Z와 입력 값 간 오차를 의미한다.
    ## 재구성 손실을 계산하기 위해선 재구성 층의 반환 값과 입력 값을 비교해야 하므로 재구성 층의 뉴런 개수는 입력 값 변수 개수 (차원크기)와 같게 설정한다.
    ## 입력배치의 차원크기가 몇 개가 될지 알 수 없으며 이는 재구성 층이 뉴런 개수 역시 알 수 없다는 뜻이다.
    ## 차원크기의 자유도를 보장하기 위해 재구성 층은 `build()` 메서드에 self.reconstruct 인스턴스 속성으로 정의한다.
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape) # 부모 클래스의 인스턴스 속성인 self.built에 True를 전달한다. 
                                         # 본 클래스의 `build()` 메서드가 부모 클래스 `build()` 메서드를 오버라이딩 하면서 self.reconstruct의 존재가 부모 클래스에 알려진다.

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)

        # 재구성 손실함수 정의
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))

        # `add_loss(x)` 메서드는 모델 실행시 손실을 저장하는 리스트에 x값을 추가한다.
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)

# %%

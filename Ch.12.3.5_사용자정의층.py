# %%
## 텐서플로 공부
## 12.3.5 사용자 정의층
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import copy
import random

exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

# %%
keras.layers.exponential_layer

# %%
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation = None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)


    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name = "kernel", shape = [batch_input_shape[-1], self.units],
            initializer = "glorot_normal"
        )

        self.bias = self.add_weight(
            name = "bias", shape = [self.units], initializer = "zeros")

        super().build(batch_input_shape)

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    # def compute_output_shape(self, batch_input_shape):
    #     return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, 
        "activation": keras.activations.serialize(self.activation)}

# %%
t = tf.Variable([[1, 2, 3], [4, 5, 6]])
t[-1]

# %%
tf.TensorShape(0)

# %%
tmp = np.ones([6, 3]) + 3
bias = np.ones([1, 3]) 

# %%
tmp + bias
# %%
tmp_layer = MyDense(units = 32, activation="relu")

# %%
tmp2 = tf.constant(tmp)
tmp2 = tf.cast(tmp2, tf.float32)
result = tmp_layer(tmp2)

# %%

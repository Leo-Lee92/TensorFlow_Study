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

# # %%
# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# from keras.utils import plot_model

# # %%
# # 데이터 불러오기
# mnist = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # 0~255 범위의 픽셀값을 0~1로 정규화
# x_train, x_test = x_train / 255.0, x_test / 255.0


# # %%
# class ANN(tf.keras.Model):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.input_layer = Dense(256, activation = 'relu', input_shape = (784, ))
#         self.hidden_layer = Dense(128, activation = 'relu')
#         self.output_layer = Dense(10, activation = 'softmax')

#     def call(self, input):
#         x = self.input_layer(input)
#         x = self.hidden_layer(x)
#         output = self.output_layer(x)
#         return output

# # %%
# from tensorflow.keras.optimizers import Adam

# MNIST_Classifier = ANN()
# cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
# optimizer = Adam(1e-4)

# # %%
# num_epochs = 1000
# batch_size = 32
# num_train_dat = x_train.shape[0]
# num_test_dat = x_test.shape[0]

# for epoch in range(num_epochs): # num_epochs회 반복해라
#     for i in range(num_train_dat // batch_size): # 배치 개수만큼 반복해라

#         # x 데이터와 라벨 데이터를 배치크기로 나눠준다
#         x_batch = x_train[i * batch_size : (i + 1) * batch_size]
#         y_batch = y_train[i * batch_size : (i + 1) * batch_size]

#         # 원래 x_batch가 (batch_size, 28, 28) 차원의 텐서였는데, reshape하여 (batch_size, 28 * 28) 차원으로 변환하라
#         # reshape()의 어떤 한 차원에서 차원크기가 결정되면 (28 * 28 ), -1을 통해 나머지 차원 크기를 자동화할 수 있다.
#         x_batch = x_batch.reshape(-1, 28 * 28) 

#         # 원래 y_batch는 (batch_size, ) 차원의 텐서이다.
#         # 우리는 이를 (batch_size, 10) 차원의 원핫 인코딩 행렬로 변환해야 한다.
#         y_batch = tf.one_hot(y_batch, 10)

# # %%
# model_params = MNIST_Classifier.trainable_variables
# with tf.GradientTape() as tape: # with구문을 통해 tf.GradientTape()를 tape라는 이름으로 객체화하여 실행한다. 에러가 있을시
#     preds = model(x_batch)
#     losses = cross_entropy(preds, y_batch)

# grads = tape.gradient(losses, model_params)
# optimizer.apply_gradients(zip(grads, model_params))

# # %%
# MNIST_Classifier(x_train, y_train) 
# plot_model(MNIST_Classifier, show_shapes = True)

# %%
# Full Script
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# 인공신경망 모델 클래스 정의
class ANN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_layer = Dense(256, activation = 'relu', input_shape = (784, ))
        self.hidden_layer = Dense(128, activation = 'relu')
        self.output_layer = Dense(10, activation = 'softmax')

    def call(self, input):
        x = self.input_layer(input)
        x = self.hidden_layer(x)
        output = self.output_layer(x)
        return output

# 메인 함수
if __name__ == "__main__":

    # data 불러오기
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  ## data 정규화

    # 모델 정의
    MNIST_Classifier = ANN() 

    # 손실 정의
    cross_entropy = CategoricalCrossentropy(from_logits = False)

    # 옵티마이저 정의
    optimizer = Adam(1e-8)

    # 하이퍼 파라미터 정의
    num_epochs = 200
    batch_size = 32
    x_train_size = x_train.shape[0]
    x_test_size = x_test.shape[0]

    for epoch in range(num_epochs):
        print('epoch :', epoch)

        for batch in range(x_train_size // batch_size):

            x_batch = x_train[(batch) * batch_size: (batch + 1) * batch_size]
            y_batch = y_train[(batch) * batch_size: (batch + 1) * batch_size]

            x_batch = x_batch.reshape(-1, 28 * 28)
            y_batch = tf.one_hot(y_batch, 10)

            trainable_params = MNIST_Classifier.trainable_variables
            with tf.GradientTape() as tape:
                preds = MNIST_Classifier(x_batch)
                losses = cross_entropy(preds, y_batch)

            grads = tape.gradient(losses, trainable_params)
            optimizer.apply_gradients(zip(grads, trainable_params))

    
    predicted_y = MNIST_Classifier(x_test.reshape(-1, 28 * 28))

    # np.amax(predicted_y, axis = 1)

    # y_test


    




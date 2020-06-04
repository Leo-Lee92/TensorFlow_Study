# 텐서플로를 활용한 ANN 구현 및 MNIST 데이터 셋 적용

### <U>Note. 본 자료는 오렐리앙 제롱의 핸즈온 머신러닝 (2판) 및 이웅렬 외 4인의 파이썬과 케라스로 배우는 강화학습을 참고하여 작성되었습니다.</U>


MNIST에서 제공하는 의류이미지를 분류하는 인공신경망 (ANN)을 구현해보자. MNIST는 28 x 28의 낮은 해상도를 가진 다양한 흑백 의류이미지 70000장 (x)과 각 의류이미지들이 어떤 카테고리 (예. 티셔츠, 바지, ..., 운동화, 가방, 발목 부츠)에 속하는지 나타내는 10개의 라벨 데이터 (y)를 제공한다. 

학습용으로 60000장, 검증용으로 10000장의 데이터를 활용하도로 하자.

텐서플로는 아래와 같이 MNIST 데이터를 손쉽게 불러올 수 있는 코드를 제공한다. MNIST 데이터는 흑백 이미지 이며 각 필셀은 0-255값을 가지는데 이를 0-1 사이의 값으로 정규화 해준 뒤 사용하도록 한다.

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
import 

# 데이터 불러오기
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 0~255 범위의 픽셀값을 0~1로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0
```

각 흑백 의류이미지를 받아 0-9로 구성된 10개의 라벨을 예측하는 인공신경망을 구현해보겠다. 구현하는 인공신경망의 구조는 다음과 같다.

- 입력값 (batch_size, 28, 28, 1) 
- 입력층 (batch_size, 784)
- 은닉층 (batch_size, 256)
- 출력층 (batch_size, 10)

한편 텐서플로 케라스를 통해 모델을 구축할 때 다음의 절차를 따라야 한다.

1. **모델 정의:** 모델을 정의한다.
   - `__init__()` 메서드에 층을 정의한다.
   - `call()` 메서드에 정의한 층들을 통과하는 순전파를 정의한다.
2. **손실함수 정의:** 옵티마이저가 적용되는 손실함수 `loss`를 정의한다.
3. **최적화 알고리즘 정의:** 모델 옵티마이저 알고리즘인 `optimizer`를 정의한다.
4. **배치 반복 정의:** 배치크기 `batch_size`를 결정하고 배치를 따라 1-4 과정을 반복하는 반복코드를 작성한다.
5. **모델 업데이트 과정**을 구현한다.
   - **예측:** 정의한 모델에 입력 값을 넣어 예측 값을 출력한다.
   - **오차 계산:** 정의한 손실함수에 예측 값 $\hat{y}$과 라벨 $y$을 넣어 오차를 계산한다.
   - **역전파:** 손실함수 (`loss`)에 대해 추정 파라미터 ($\hat{\theta}$)의 편미분 값을 구한다.
   - **업데이트:** 구한 편미분 값과 앞서 정의한 옵티마이저 알고리즘 `optimizer`를 통해 모델 (정확히는 파라미터들)을 업데이트 한다.  

아래의 코드는 위 절차중 과정 1을 보여준다.

```python
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
```

아래의 코드는 위 절차중 과정 2-3을 보여준다.

```python
from tensorflow.keras.optimizers import Adam

MNIST_Classifier = ANN() # 정의된 모델을 MNIST_Classifier라는 인스턴스로 객체화
cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
optimizer = Adam(1e-4)
```

- 텐서플로 자체제공 클래스인 `tf.keras.losses` 에서 `CategoricalCrossentropy` 메서드를 불러와 `cross_entropy`라는 이름으로 인스턴스 객체화 하였다. `CategoricalCrossentropy` 메서드를 불러올 때 `from_logits = False` 로 인자를 세팅하였다.
- 텐서플로 자체제공 클래스인 `tf.keras.optimizers` 에서 `Adam` 메서드를 불러와 `optimizer`라는 이름으로 인스턴스 객체화 하였다. `Adam(lr)`의 인자인 학습률을 `1e-4`로 세팅하였다.

참고로 `Adam` 최적화 알고리즘은 기본적인 성능을 보장해주는 가장 대중적이고 최적화 기법으로, 자신의 모델을 업데이트하는 최적의 최적화 알고리즘이 무엇인지 모를 때 사용해주면된다. 그 이유는 아래의 [그림 1]에 나타난 바와 같이 `Adam`은 보폭 방향 (기울기), 배치, 보폭 크기(학습률), 이전 파라미터를 모두 고려하가 때문이다.

<p align = "center"><img src = "https://user-images.githubusercontent.com/61273017/83608687-36ca1400-a5b8-11ea-8305-a6e838561d97.png" weight = 400 height = 300></p>
<p align = "center">[그림 1] 옵티마이저 맵</p>

아래의 코드는 위 절차중 과정 4를 보여준다.

```python
for epoch in range(num_epochs): # num_epochs회 반복해라
    for i in range(num_train_dat // batch_size): # 배치 개수만큼 반복해라

        # x 데이터와 라벨 데이터를 배치크기로 나눠준다
        x_batch = x_train[i * batch_size : (i + 1) * batch_size]
        y_batch = y_train[i * batch_size : (i + 1) * batch_size]

        # 원래 x_batch가 (batch_size, 28, 28) 차원의 텐서였는데, reshape하여 (batch_size, 28 * 28) 차원으로 변환하라
        # reshape()의 어떤 한 차원에서 차원크기가 결정되면 (28 * 28 ), -1을 통해 나머지 차원 크기를 자동화할 수 있다.
        x_batch = x_batch.reshape(-1, 28 * 28) 

        # 원래 y_batch는 (batch_size, ) 차원의 텐서이다.
        # 우리는 이를 (batch_size, 10) 차원의 원핫 인코딩 행렬로 변환해야 한다.
        y_batch = tf.one_hot(y_batch, 10)
```

아래의 코드는 위 절차중 과정 5를 보여준다.

```python
model_params = MNIST_Classifier.trainable_variables
with tf.GradientTape() as tape: 
    preds = model(x_batch)  # 과정 5의 예측
    losses = cross_entropy(preds, y_batch) # 과정 5의 오차 계산

grads = tape.gradient(losses, model_params) # 과정 5의 역전파
optimizer.apply_gradients(zip(grads, model_params)) # 과정 5의 업데이트
```
- `with A as a: [block]` 구문은 일명 context manager로 파이썬에 내장된 문법이다. `A` 메서드는 `a`라는 이름의 인스턴스로 실행되며 `[block]` 내에서 실행된 모든 연산은 `a`에 기록된다. 
  - `[block]` 내 연산이 실행될 때 호출되는 필수 메모리영역, 레지스터 값 등의 리소스를 총칭하여 context라 부른다. 즉 컨텍스트는 `[block]` 안에서 실행되는 연산이 기록되는 공간이다.
  - `with` 구문은 내부적으로 context mangager 프로토콜을 준수한 "컨텍스트 매니저" 객체만을 활용할 수 있다. "컨텍스트 매니저"란 `__enter__()`, `__exit__()` 메서드가 구현된 객체 (클래스, 함수 등)를 의미한다. `with` 구문은 입력으로 받은 컨텍스트 매니저 객체의 종료 조건이 달성되거나 예외처리가 발생하면 객체의 실행을 종료함과 동시에 자동으로 리소스를 릴리즈 (release)하여 resource leak을 방지한다.
- 위 코드는 텐서플로에서 제공하는 `tf.GradientTape()` 라는 컨텍스트 매니저 객체를 `tape` 라는 이름의 인스턴스로 정의하여 `with` 구문에서 실행한다. 이 때 `[block]` 안에서 실행되는 연산들은 `tape` 컨텍스트 매니저 객체에 할당된 컨텍스트에 기록된다.
- `tf.GradientTape.gradient()` 메서드를 실행하면 `__exit__` 메서드가 호출되어 `tf.GradientTape()` 가 할당받은 모든 컨텍스트를 릴리즈한다. 즉 `tape` 객체가 할당받은 모든 컨텍스트들이 릴리즈 된다. 이는 업데이트 과정을 수행하기 위해선 매 반복마다 오차와 편미분값을 계산해야 하는데, 업데이트 과정 수행후 컨텍스트를 릴리즈해주지 않으면 자칫 엄청난 resource leak가 발생할 수 있기 때문이다. 
  - 학습구조 상 동일 gradient를 여러번 재활용해야 한다면 컨텍스트 릴리즈 되지 않도록 `tf.GradientTape(persistent = True)` 를 설정하면 된다.
- `preds = model(x_batch)` : 과정 1에서 정의한 모델을 통해 값을 예측한다.
- `losses = cross_entropy(preds, y_batch)` : 과정 2에서 정의한 손실함수를 통해 오차를 계산한다.
- `grads = tape.graident(losses, model_params)` : 오차 (losses)의 학습가능한 매개변수 (model_params)에 대한 편미분 값 (grads)을 계산한다. 
- `optimizer.apply_gradients(zip(grads, model_params))` : 위에서 계산한 grads을 과정 3에서 정의한 최적화 알고리즘에 활용하여 model_params(t)을 model_params(t+1)로 업데이트 한다. 
  - `zip()`은 동일한 인덱스 크기를 가진 리스트를 column-wise stacking해주는 역할을 한다. 즉  `optimizer.apply_gradients()` 메서드는 grads와 model_params 리스트를 column-wise stacking 한 데이터를 요구한다고 보면 된다.

MNIST 흑백 의류이미지를 분류하는 CNN 망을 학습하는 전체 코드는 아래와 같이 작성하면 된다.

```python
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
    num_epochs = 20
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

        print('loss :', losses)
```
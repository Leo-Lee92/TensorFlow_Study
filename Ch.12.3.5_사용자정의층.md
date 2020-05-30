# 12.3.5 사용자 정의 층 __(Layer)__

### <U>Note. 본 자료는 오렐리앙 제롱의 핸즈온 머신러닝 (2판)을 참고하여 작성되었습니다.</U>

신경망 모델을 개발하다보면 텐서플로 케라스가 제공하지 않는 사용자 정의층을 가진 네트워크를 만들어야 할 때가 있다.

사용자 정의층을 어떻게 만드는지 알아보자.

## 1. keras.layers.Lambda 기반 (가중치가 없는 층)

단순히 맵핑, 차원변환 등의 역할을 수행하는 구조변환 (ex. `keras.layers.Flatten`) 및 활성화 층 (ex. `keras.layers.ReLU`) 등은 __가중치가 존재하지 않는__ 층이다. 이러한 층들을 사용자 정의층으로 만들고자 한다고 가정해보자.

가장 간단한 방법은 파이썬 함수를 만든 후 `keras.layers.Lambda` 층으로 감싸는 것이다.다음은 입력값에 지수함수가 적용되는 활성화 층을 정의한 코드이다.

```python
import keras
exponential = keras.layers.Lambda(lambda x: tf.exp(x))
```

위에서 정의한 exponential 층은 케라스 자체제공 층들과 동일하게 `keras.layers.exponential`로 호출할 수 있다. 또한, 이렇게 정의된 층은 시퀀셜 API, 함수형 API,서브클래싱 API에서 보통의 층과 동일하게 사용가능하다.

## 2. 상위 클래스 상속 기반 (가중치가 있는 층)

맵핑, 차원변환 층과 달리 `Dense`, `Conv`, `SimpleRNN` 등은 신경망의 학습이 이뤄지는 단일 층으로, 해당 층들은 오차역전파를 통해 학습정보를 저장하고 업데이트하는 가중치 배열을 가지고 있다. 만약 이러한 가중치가 존재하는 층을 만들고자 한다면 `keras.layers.Layer`를 상속해야 한다.
`
아래의 코드는 Dense 층의 간소화 버전을 구현한 클래스이다.

```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation = None, **kwargs):
        super().__init__(**kwargs)
        self.units - units
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

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, 
        "activation": keras.activations.serialize(self.activation)}
```
코드를 하나씩 살펴보겠다.

- 사용자 정의 클래스인 `class MyDense(keras.layers.Layer)`가 `keras.layers.Layer`를 인자로 받음으로써 `keras.layers.Layer` 클래스를 상속받게 된다.
  - 부모 클래스로는 보통 텐서플로 자체제공 클래스를 사용한다.  
- 생성자 (`__init__`) 는 사용자 입력 파라미터 등 하이퍼 파라미터를 매개변수로 받는다.
- 생성자에서 클래스 작동에 필요한 각종 매개변수들이 초기화 된다.
  - 위 코드에서 하이퍼 파라미터는 `units`와 `activation`이다.
  - 생성자는 부모 클래스 생성자에 정의된 인자 (매개변수)들도 상속받는다. 텐서플로 자체제공 클래스의 생성자에 정의된 인자들은 `dict` 자료형이다. 즉 사용자 정의 클래스가 상속할 부모 클래스가 텐서플로 자체제공 클래스라면  `dict` 자료형 인자들을 상속해야 한다. `dict` 자료형 인자들을 상속하기 위해 사용자 정의 클래스의 생성자에 `**kwargs` 인자를 정의해야 한다. 
    - `**kwargs`를 통해 input_shape, trainable, name과 같은 필수 매개변수들을 부모 클래스로부터 직접 상속하여 사용자 정의 클래스에서 처리할 수 있게 된다. 부모 클래스가 텐서플로 자체제공 클래스라면 위의 필수 매개변수들의 자료형은 보통 Tensor 타입이다. 
    - 매개변수들은 `self.units` 와 같은 인스턴스 속성으로 저장된다. `self` 는 인스턴스화 된 객체의 이름을 상속한다.
    - `activation` 매개변수의 인자의 기본값은 `None` 으로 강제되어 있다. 사용자 정의 클래스를 실행할 때 `activation = "relu"`와 같은 문자열을 입력한다. 해당 문자열은 텐서플로의 자체제공 클래스인 `keras.activation` 클래스에 정의된 `get()` 메서드의 인자로 전달된다. 전달된 문자열에 상응하는 객체가 반환되어 인스턴스 속성 `self.activation` 으로 저장된다.
      - (참고) `keras.activations.get()` 대신 `keras.layers.Activation()` 을 써도 된다.
      - (참고) `tuple` 자료형 인자를 상속받고 싶은 경우 클래스의 생성자에 `*args` 인자가 정의되어 있어야 한다.
  
- `__init__()` 메서드 (생성자)는 __외장형 인스턴스 속성__ (예. `self.units = units`)을 정의한다. __외장형 인스턴스 속성__ 이란 사용자 정의 클래스의 작동에 필요한 변수들로 (1) 사용자가 직접 값을 입력하거나 (2) 상위 클래스로부터 상속받은 속성들이다. 즉 클래스 외부로부터 클래스로 전달되어 정의되는 속성을 뜻한다. 
- `build()` 메서드는 __내장형 인스턴스 속성__ (예. `self.kernel = self.add_weight()`)을 정의한다. __내장형 인스턴스 속성__ 이란 사용자 정의 클래스의 작동에 필요한 세부적인 요소들을 직접 정의하여 변수화한 것이다. 즉 클래스 내부에서 정의되는 속성을 뜻한다. 
  - `build()` 메서드의 경우 사용자 정의 층이 최초로 사용될 때 `call()` 메서드에 앞서 무조건 호출된다. 층 사용 초기에 호출되어 필요한 내장형 인스턴스 속성들을 정의해두는 것이다.
  - `build()`는 batch_input_shape (= input_shape)라는 클래스 속성을 인자로 받는다. 사용자 정의 층을 활용할 때 입력값을 층으로 전달하면 사용자 정의층이 상속하는 최상위 부모 클래스인 keras 모듈에 의해 batch_input_shape가 자동으로 결정된다. 즉, `build(self, batch_input_shape)`를 사용하면 층마다 input, output 차원을 직접 정의하지 않아도 된다는 장점이 있다. 
- `call()` 메서드는 사전 인스턴스 속성과 자체 인스턴스 속성을 활용하여 사용자 정의 클래스가 수행해야 할 연산과정 (예. `self.activation(X @ self.kernel + self.bias)`), 즉 프로시져를 정의한다.
- `compute_output_shape()` 메서드는 사용자 정의 층의 출력크기를 반환한다. 이 예에서 마지막 층은 뉴런의 개수를 차원크기로 가진다. 각 배치입력이 특정 층에 입력될 때의 차원크기는 `tf.TensorShape([, ])` 자료형으로 처리된다. 배치크기가 6이며 변수의 개수 (차원크기)가 3일 때 `batch_input_shape`는 `tf.TensorShape([6, 3])` 형태가 된다. TensorShape 자료형은 `batch_input_shape.as_list()` 를 통해 List 자료형인 `[6, 3]`으로 변환가능하다. `batch_input_shape.as_list()[:-1]`는 리스트 `[6, 3]`에서 마지막 열을 제외한 리스트 `[6]`을 반환하고 `self.units = 4`라고 가정하면, `bach_input_shape.as_list()[:-1] + [self.units]`는 리스트 덧셈인 `[6] + [4]` 을 통해 `[6, 4]`가 된다. 즉 마지막 층의 차원크기는 배치크기 (6) x 마지막 층 차원 (4)을 반환하라는 뜻이다.  
- `get_config()` 메서드는 부모 클래스에서 정의된 하이퍼 파라미터 리스트를 반환하여 base_config에 저장한 뒤, 파이썬 3.5 버전의 리스트 요소추가 간편문법인 {**list, "key": value}를 사용하여 사용자 정의 클래스에서 정의된 하이퍼 파라미터를 부모 클래스의 하이퍼 파라미터 리스크에 추가한다.

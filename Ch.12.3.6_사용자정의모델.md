# 12.3.6 사용자 정의 모델 __(Model)__

### <U>Note. 본 자료는 오렐리앙 제롱의 핸즈온 머신러닝 (2판)을 참고하여 작성되었습니다.</U>

사용자 정의 모델 역시 사용자 정의 층과 마찬가지로 텐서플로 자체제공 부모 클래를 상속하는 **서브클래싱 API** 형식으로 작성된다.

방법은 간단하다. 사용자 정의 클래스에 `keras.Model`을 상속시켜 사용자 정의 클래스가 텐서플로에서 자체제공하는 신경망 모델에 필요한 필수 매개변수들을 처리할 수 있게 만든다. 이후 사용자 정의 클래스의 `__init__` (생성자)에서 필요한 층과 변수를 정의하고 `call()` 메서드에 순전파 (propagation) 과정을 구현한다. 아래의 [그림 1]과 같은 모델을 만든다고 가정해보자.

<p align = "center"><img src = "https://user-images.githubusercontent.com/61273017/83327571-d2593d00-a2b7-11ea-80a2-fbd0bfcd519a.png" width = "500" height = "400"></p>
<p align = "center"> [그림 1] 사용자 정의 모델: 스킵 연결이 있는 사용자 정의 잔차 블록(ResidualBlock) 층을 가진 예제 모델 </p>

사용자 정의 모델은 Dense - ResidualBlock1 - ResidualBlock2 - Dense로 구축되어있다. ResidualBlock층은 두 개의 완전 연결층 (Dense)과 스킵 연결로 구성되어 있다. 모델의 작동순서는 다음과 같다.
- 입력값이 완전 연결층 (Dense)을 통과하여 중간값으로 반환된다.
- 중간값이 첫번째 잔차블록층 (ResidualBlock1)을 세 번 통과한다.
  - 첫번째 잔차블록층은 서브레이어로서 연속된 2개의 완전 연결층과 스킵연결 과정으로 구성되어 있다. 
- 중간값이 두번째 잔차블록층 (ResidualBlock2)을 통과한다.
- 중간값이 완전 연결층 (Dense)을 통과하여 출력값으로 반환된다.

아래의 코드는 ResidualBlock 층을 만드는 코드이다.

```python
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers,  n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation = "elu", kernel_initalizer = "he_normal") for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z
```
코드에 대한 해설은 다음과 같다.
- 부모 클래스로 `keras.layers.Layer` 를 상속한다.
- **`__init__()`은 층의 필수 매개변수들을 초기화** 한다. 
  - `super()` 를 통해 부모 클래스의 `__init__()` 에 정의된 인스턴스 속성, 즉 매개변수들 (`**kwargs`)을 상속받아 `ResidualBlock()` 클래스의 인스턴스 속성으로 초기화 한다. 
    - 해당 과정이 없으면 `keras.layers.Layer` 클래스를 상속하더라도 인스턴스 속성의 상속이 명시화되지 않아 `ResidualBlock()` 클래스에서 사용할 수 없다
  - `keras.layers` 클래스의 `Dense` 메소드를 `n_layers` 번 호출하여 반환되는 `n_layers` 개의 객체를 `[x in for _ in range(n_layers)]`를 통해 `List` 에 담아 `self.hidden` 이라는 인스턴스 속성으로 정의한다 (인스턴스화 한다). `self.hidden`은 `n_layers`개의 완전 연결 은닉층이 적층된 구조로 `ResidualBlock()` 층을 구성하는 서브레이어이다. 
    - `keras.layers.Dense` 로부터 호출되는 각 객체는 가중치 배열, 차원크기, dtype 등의 정보를 담고 있는 Tensor 자료형 객체이다.
    - 각 `Dense()` 가 반환하는 객체인 은닉층의 <U>(1) 뉴런 개수는 `n_neurons` 개</U>, <U>(2) 가중치 배열의 초기화 방법은  `"he_normal"`</U>, <U>(3) 활성화 함수는 `"elu"`로 설정하였다.</U> 
- **`call()` 메서드는** 생성자에서 정의된 인스턴스 속성들과 입력값 `inputs`을 활용하여 `ResidualBlock()` 층의 **순전파 프로시져를 정의**한다.
  - `self.hidden` 에는 `n_layers`의 객체가 `List` 형태로 담겨 있으므로 `for layer in self.hidden:` 구문은 개별 은닉층을 정의하는 `Dense()`가 반환하는 객체들을 차례로 호출함으로써 `ResidualBlock()` 층의 **서브레이어의 호출과정을 정의**한다. 차례로 호출된 은닉층들은 layer라는 변수에 저장된다.
  - `Z = layer(Z)`는 입력값 `Z`가 각 은닉층을 통과하며 `n_neurons` 의 차원을 가지도록 재구축/표현 (represenation) 된 잠재행렬을 `Z`로 재정의함을 명시한다. 즉 초기 입력값 `Z`가 앞서 정의된 서브구조의 호출과정을 따라 복수의 층들을 통과하며 재구축되는 **순전파 과정을 정의**한다.
- 마지막의 `inputs + Z`는 초기 입력값과 마지막 서브레이어를 통과하여 재구축된 `Z` 를 단순히 더함으로써 `ResidualBlock()` 층의 스킵연결을 정의한다.

위에서 `ResidualBlock()` 층을 정의하였으므로 이제 `ResidualBlock()` 층을 포함한 신경망 모델을 정의해보겠다. 신경망 모델의 이름은 잔차회귀 `ResidualRegressor`이며 아래의 코드와 같이 정의된다.

실험 ㅋ

# 12.3.6 사용자 정의 모델 __(Model)__

### <U>Note. 본 자료는 오렐리앙 제롱의 핸즈온 머신러닝 (2판)을 참고하여 작성되었습니다.</U>

사용자 정의 모델 역시 사용자 정의 층과 마찬가지로 텐서플로 자체제공 부모 클래를 상속하는 **서브클래싱 API** 형식으로 작성된다.

방법은 간단하다. 사용자 정의 클래스에 `keras.Model`을 상속시켜 사용자 정의 클래스가 텐서플로에서 자체제공하는 신경망 모델에 필요한 필수 매개변수들을 처리할 수 있게 만든다. 이후 사용자 정의 클래스의 `__init__` (생성자)에서 필요한 층과 변수를 정의하고 `call()` 메서드에 순전파 (propagation) 과정을 구현한다. 아래의 [그림 1]과 같은 모델을 만든다고 가정해보자.

<p text-align = "center"> <img src = "https://user-images.githubusercontent.com/61273017/83327571-d2593d00-a2b7-11ea-80a2-fbd0bfcd519a.png" width = "500" height = "400"></p>
<center> [그림 1] 사용자 정의 모델: 스킵 연결이 있는 사용자 정의 잔차 블록(ResidualBlock) 층을 가진 예제 모델 </center>

사용자 정의 모델은 Dense - ResidualBlock1 - ResidualBlock2 - Dense로 구축되어있다. ResidualBlock층은 두 개의 완전 연결층 (Dense)과 스킵 연결로 구성되어 있다. 모델의 작동순서는 다음과 같다.
- 입력값이 완전 연결층 (Dense)을 통과하여 중간값으로 반환된다.
- 중간값이 첫번째 잔차블록층 (ResidualBlock1)을 세 번 통과한다.
- 중간값이 두번째 잔차블록층 (ResidualBlock2)을 통과한다.
- 중간값이 완전 연결층 (Dense)을 통과하여 출력값으로 반환된다.

아래의 코드는 ResidualBlock 층을 만드는 코드이다.
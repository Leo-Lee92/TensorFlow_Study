# 12.3.7 모델 구성 요소에 기반한 손실과 지표 __(add_loss, add_metric))__

### <U>Note. 본 자료는 오렐리앙 제롱의 핸즈온 머신러닝 (2판)을 참고하여 작성되었습니다.</U>

일반적으로 손실이나 지표는 실제 $y$값 (정답; 입력된 정답)과 예측 $\hat{y}$값 (추정; 출력된 예측)으로부터 얻은 오차에 기반하여 정의된다. 실제 $y$값과 예측 $\hat{y}$값 간의 차, 즉 오차를 계산하는 방식을 정의한 함수를 손실함수라고 한다. 손실함수는 `mean_squared_error`, `categorical_crossentropy` 등 다양한 형태로 존재한다. 여기서 주의할 점은 우리가 보통 손실을 정의할 때 실제 값과 예측 값의 차를 활용한다는 것이다.

그러나 때로는 실제 값 (입력 값)과 예측 값 (출력 값)이 아닌 중간 값 (은닉층의 가중치 혹은 중간 활성화 층의 반환 값) 등과 같이 모델의 구성 요소를 기반으로 손실을 정의해야 할 때가 있다. 이렇게 중간 값


<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a>
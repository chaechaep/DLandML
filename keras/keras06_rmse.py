# 1. 데이터 구성
import numpy as np
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array(range(11,21))
y_test = np.array(range(11,21))


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(4))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)

#rmse(평균 제곱근 편차) 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))





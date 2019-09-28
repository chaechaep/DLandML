
# m03_xor.py를 keras로 리폼
import numpy as np
# 1. data
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0,1,1,0])
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=100, batch_size=1)
loss, acc = model.evaluate(x_data, y_data)
print("acc : ", acc)

x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_predict = model.predict(x_test)


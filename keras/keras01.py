import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)
loss, acc = model.evaluate(x, y)
print("acc : ", acc)

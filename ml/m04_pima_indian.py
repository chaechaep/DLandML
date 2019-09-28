from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# data load
dataset = np.loadtxt('./data/csv/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

# model configure
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model fit
model.fit(X, Y, epochs=200, batch_size=10)

# print result
print("\n Accuracy: %.4f" %(model.evaluate(X, Y)[1]))
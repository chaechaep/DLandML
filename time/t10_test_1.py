from numpy import array
from numpy import hstack

from keras.models import Sequential, Model
from keras.layers import Dense, concatenate

X1 = array([1, 2, 3])
y1 = array([1, 2, 3])
X1 = X1.reshape((3,1))
y1 = y1.reshape((3,1))

X2 = array([3, 4, 5])
y2 = array([3, 4 ,5])
X2 = X2.reshape((3,1))
y2 = y2.reshape((3,1))

seq_model1 = Sequential()
seq_model1.add(Dense(50, input_dim=1, activation='relu'))
seq_model1.add(Dense(1))

seq_model2 = Sequential()
seq_model2.add(Dense(50, input_dim=1, activation='relu'))
seq_model2.add(Dense(1))

merge1 = concatenate([seq_model1, seq_model2], axis=0)
output = Dense(2)(merge1)

model = Model(inputs=[seq_model1, seq_model2], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

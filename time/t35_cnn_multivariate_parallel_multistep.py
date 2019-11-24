from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                 input_shape=(3, 2)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')

model1 = Sequential()
model1.add(LSTM(64, input_shape=(3, 1)))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(2))
model1.compile(optimizer='adam', loss='mse')

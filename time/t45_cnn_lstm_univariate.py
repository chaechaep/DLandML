from keras import Sequential
from keras.layers import LSTM, Dense, Flatten, Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from numpy import array
from keras.layers import TimeDistributed

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) -1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps = 4

X, y = split_sequence(raw_seq, n_steps)

# n_shape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
print(X.shape)
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
print(X.shape)


# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                          input_shape=(2, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# model.fit(X, y, epochs=200, verbose=1)
#
# x_input = array([60, 70, 80, 90])
# x_input = x_input.reshape((1, n_seq, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)

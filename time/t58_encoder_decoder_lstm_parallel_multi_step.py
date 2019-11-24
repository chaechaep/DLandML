from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, LSTM, TimeDistributed
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix: out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)
n_steps_in, n_steps_out = 3, 2

X, y = split_sequence(dataset, n_steps_in, n_steps_out)

n_features = X.shape[2]
# x_shape = X.shape[1]*X.shape[2]
# y_shape = y.shape[1]*y.shape[2]
# X = X.reshape((X.shape[0], x_shape, n_features))
# y = y.reshape((y.shape[0], y_shape, n_features))

print(X.shape, y.shape)
for i in range(len(X)):
    print(X[i], y[i])

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=300, verbose=1)

x_input = array([[80, 85, 165], [90, 95, 185], [100, 105, 205]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input)
print(yhat)
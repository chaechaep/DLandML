from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D
from numpy import array
from numpy import hstack

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
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

n_steps=3
X, y = split_sequence(dataset, n_steps=n_steps)
print(X.shape, y.shape)

for i in range(len(X)):
    print(X[i], y[i])
n_features=X.shape[2]
# define model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',
                 input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200)

x_input = array([[[80, 85], [90, 95], [100, 105]], [[90, 95], [100, 105], [110, 115]]])
yhat = model.predict(x_input)
print(yhat)



from keras.layers import Dense, concatenate, Input
from numpy import array, hstack
from keras.models import Sequential, Model

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix :out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(dataset, n_steps_in, n_steps_out)

print(X.shape, y.shape)
for i in range(len(X)):
    print(X[i], y[i])

n_input = X.shape[1]*X.shape[2]
X = X.reshape(X.shape[0], n_input)

n_output = y.shape[1]*y.shape[2]
y = y.reshape(y.shape[0], n_output)

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(500))
model.add(Dense(340, activation='relu'))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# fit model
model.fit(X, y, epochs=500) #, verbose=0)
#
# # demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_input))
#
y_pred = model.predict(x_input)
print(y_pred)
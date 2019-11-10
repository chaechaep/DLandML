from keras.layers import Dense
from numpy import array, hstack
from keras.models import Sequential

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, : -1], sequence[end_ix-1 :out_end_ix, -1]
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


# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# fit model
model.fit(X, y, epochs=2000) #, verbose=0)
#
# # demonstrate prediction
x_input = array([70, 80, 90, 75, 85, 98])
x_input = x_input.reshape((1, n_input))
#
y_pred = model.predict(x_input)
print(y_pred)

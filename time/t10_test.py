# dim이 1인 아웃풋 모델 3개를 작성
# t08로 만들것


from numpy import array
from numpy import hstack
from keras.models import Sequential, Model
from keras.layers import Dense,Input
from keras.layers.merge import concatenate

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[i+1:end_ix+1, :]
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
n_steps = 3
X, y = split_sequence(dataset, n_steps=n_steps)
print(X.shape, y.shape)

for i in range(len(X)):
    print(X[i], y[i])

# flatten
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))

# model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=2000, verbose=0)

x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
yhat = yhat.reshape((3,3))
print(yhat)
"""
#
X1 = X[:, :, 0]
X2 = X[:, :, 1]
X3 = X[:, :, 2]

# first input model
visible1 = Input(shape=(n_steps,))
dense1 = Dense(100, activation='relu')(visible1)
# second input model
visible2 = Input(shape=(n_steps,))
dense2 = Dense(1000, activation='relu')(visible2)
# third input model
visible3 = Input(shape=(n_steps,))
dense3 = Dense(1000, activation='relu')(visible3)

# merge input models
merge = concatenate([dense1, dense2, dense3])
output = Dense(3)(merge)
model = Model(inputs=[visible1, visible2, visible3], outputs=output)
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit([X1, X2, X3], y, epochs=2000, verbose=0)

# demonstrate prediction
x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
x1 = x_input[:, 0].reshape((1, n_steps))
x2 = x_input[:, 1].reshape((1, n_steps))
x3 = x_input[:, 2].reshape((1, n_steps))
yhat = model.predict([x1, x2, x3], verbose=0)

print(yhat)
"""
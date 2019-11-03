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
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
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

#

n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))
visible = Input(shape=(n_input,))
dense = Dense(100, activation='relu')(visible)

output1 = Dense(1)(dense)
output2 = Dense(1)(dense)
output3 = Dense(1)(dense)
model = Model(inputs=visible, outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, [y1, y2, y3], epochs=2000, verbose=0)

# demonstrate prediction
x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)

print(yhat)

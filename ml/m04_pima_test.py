from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# data load
dataset = np.loadtxt('./data/csv/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=66, train_size=0.7)
# model configure
# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
model = RandomForestClassifier()

# model compile
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model fit
# model.fit(X, Y, epochs=200, batch_size=10)
model.fit(x_train, y_train)

# print result
# print("\n Accuracy: %.4f" %(model.evaluate(X, Y)[1]))

y_predict = model.predict(x_test)
print("Accuracy Score : ", accuracy_score(y_test, y_predict))

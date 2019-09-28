import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# data load
iris_data = pd.read_csv('./data/csv/iris.csv', encoding='utf-8',
                        names=['a', 'b', 'c', 'd', 'y'])

print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# x, y split
y = iris_data.loc[:, "y"]
x = iris_data.loc[:, ['a', 'b', 'c', 'd']]

from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
e.fit(y)
y1 = e.transform(y)

from keras.utils import np_utils
# one-hot encoding
y_encoded = np_utils.to_categorical(y1)

print("================================")
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, train_size=0.7, shuffle=True)

# # model configure & fit
# model = SVC()
# model.fit(x_train, y_train)
model = Sequential()
model.add(Dense(15, input_dim=4, activation='relu'))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=1)

# evaluate
print("\n Accuracy: %.4f" %(model.evaluate(x_test, y_test)[1]))
y_pred = model.predict(x_test)

# print("Accuracy Score:", accuracy_score(y_test, y_pred))


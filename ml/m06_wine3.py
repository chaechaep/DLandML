import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# data load
wine = pd.read_csv('./data/csv/winequality-white.csv', sep=";", encoding="utf-8")

# x, y split
y = wine["quality"]
x = wine.drop("quality", axis=1)

newlist = []
for v in list(y):
    if v <=4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist
from keras.utils import np_utils
# one-hot encoding
y_encoded = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=30)

# fit
# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# aaa = model.score(x_test, y_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(200, input_dim=11, activation='relu'))
model.add(Dense(150))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=500)

print("\n Accuracy: %.4f" %(model.evaluate(x_test, y_test)[1]))

# evaluate
y_pred = model.predict(x_test)
print(y_pred)
# print(classification_report(y_test, y_pred))
# print("Accuracy score : ", accuracy_score(y_test, y_pred))
# print(aaa)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# data load
iris_data = pd.read_csv('./data/csv/iris.csv', encoding='utf-8',
                        names=['a', 'b', 'c', 'd', 'y'])

print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# x, y split
y = iris_data.loc[:, "y"]
x = iris_data.loc[:, ['a', 'b', 'c', 'd']]

print("================================")
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7, shuffle=True)

# model configure & fit
model = SVC()
model.fit(x_train, y_train)

# evaluate
y_pred = model.predict(x_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))


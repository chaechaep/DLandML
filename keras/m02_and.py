from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0,0,0,1]

# 2. model
# model = LinearSVC()
model = SVC()

# 3. fit
model.fit(x_data, y_data)

# 4. evaluate, predict
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

print(y_predict)
print(accuracy_score([0, 0, 0, 1], y_predict))
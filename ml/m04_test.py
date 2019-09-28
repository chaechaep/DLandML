from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = x.reshape(-1, 1)
# y = y.reshape(-1, 1)

model = LinearSVC()
model.fit(x, y)

# 4. evaluate, predict
x_test = np.array([1,2,3,4,5])
x_test = x_test.reshape(-1, 1)

y_predict = model.predict(x_test)

print(y_predict)
print(accuracy_score([1,2,3,4,5], y_predict))
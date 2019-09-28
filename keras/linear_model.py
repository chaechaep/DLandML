from sklearn import linear_model
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([6,7,8,9,10])

model = linear_model.LinearRegression()
model.fit(x, y)

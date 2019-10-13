from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression()

score = cross_val_score(model, iris.data, iris.target, cv= 5)

print("교차 검증 점수 : ", score)
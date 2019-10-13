import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import RandomizedSearchCV

iris_data = pd.read_csv("./data/csv/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    train_size=0.8, shuffle=True)

parameters = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [10, 20, 30, 50],
    'n_jobs': [-1]
}

clf = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True)
clf.fit(x_train, y_train)
print("최적의 매개 변수 = ", clf.best_estimator_)

y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
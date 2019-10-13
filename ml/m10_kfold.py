import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators


iris_data = pd.read_csv("./data/csv/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# warnings.filterwarnings('ignore')
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
#                                                     train_size=0.8, shuffle=True)

kfold_cv = KFold(n_splits=5, shuffle=True)
# classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    if hasattr(model, "score"):
        scores = cross_val_score(model, x, y, cv=kfold_cv)
        print(name, "의 정답률 = ")
        print(scores)
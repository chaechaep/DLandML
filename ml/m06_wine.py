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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# fit
model = RandomForestClassifier()
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# evaluate
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("Accuracy score : ", accuracy_score(y_test, y_pred))
print(aaa)
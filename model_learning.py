import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = sns.load_dataset('iris')


cols = iris.columns.tolist()
X = iris[cols[:-1]]
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
model = RandomForestClassifier(n_estimators = 10000).fit(X_train, y_train)
print(model.score(X_test, y_test))

with open('iris_predictor.pickle', 'wb') as f:
    pickle.dump(model, f)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle

data = load_iris(as_frame=True)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42, 
                                                    stratify=y)

X_reduced = PCA(n_components=3)
X_reduced.fit(X_train)
X_train_new = X_reduced.transform(X_train)
X_test_new = X_reduced.transform(X_test)

model = LogisticRegression()
model.fit(X_train_new, y_train)


def save(model, X_test, y_test):
    with open('model.pickle', 'wb') as file:
        pickle.dump(model, file)
    with open('X_test.pickle', 'wb') as file:
            pickle.dump(X_test, file)
    with open('y_test.pickle', 'wb') as file:
            pickle.dump(y_test, file)

        
save(model, X_test_new, y_test)
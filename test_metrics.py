import pytest
import pickle
from sklearn.metrics import r2_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

@pytest.fixture
def min_score():
    return 0.95


def load():
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('X_test.pickle', 'rb') as file:
        X_test = pickle.load(file)
    with open('y_test.pickle', 'rb') as file:
        y_test = pickle.load(file)
    return model, X_test, y_test


def calculate_score():
    model, X_test, y_test = load()
    return r2_score(y_test, model.predict(X_test))


def test_score(min_score):
    assert calculate_score() >= min_score
    

def test_metrics():
    
    model, X_test, y_test = load()
    acc = classification_report(y_test, model.predict(X_test))
    
    with open("metrics.txt", "w") as outfile:
        outfile.write(acc)
    
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, normalize="true", cmap=plt.cm.Blues
    )
    plt.savefig("plot.png")
    
    assert 1 == 1

import pickle


def load():
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('X_test.pickle', 'rb') as file:
        X_test = pickle.load(file)
    return model, X_test


def save_predict(y_pred):
    with open('predict.txt', 'w') as output:
        output.write(str(y_pred))

model, X_test = load()

y_pred = model.predict(X_test)

save_predict(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, X_test_scaled, y_test):

    # prediction
    y_pred = model.predict(X_test_scaled)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, confusion_mat

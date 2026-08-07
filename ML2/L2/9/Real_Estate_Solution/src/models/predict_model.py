# Evaluate modelss
from sklearn.metrics import mean_absolute_error

# function to predict and evalute
def evaluate_model(model, X_test, y_test):
    
    # prediction
    y_pred = model.predict(X_test)

    # calculate mae
    test_mae = mean_absolute_error(y_pred, y_test)

    return test_mae

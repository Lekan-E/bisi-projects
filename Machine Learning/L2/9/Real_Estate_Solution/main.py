from src.data.make_dataset import *
from src.features.build_features import *
from src.models.train_model import *
from src.models.predict_model import *

if __name__ == "__main__":

    # laod and preprocess the data
    datapath = 'data/raw/final.csv'
    df = load_and__preprocess_data(datapath)

    # create x,y variables
    X, y = create_var(df)

    # train, test split
    LRmodel, X_test_lr, y_test_lr = train_LRmodel(X,y)
    RFmodel, X_test_rf, y_test_rf = train_RFmodel(X,y)

    # Evaluate model
    lr_mae = evaluate_model(LRmodel, X_test_lr, y_test_lr)
    rf_mae = evaluate_model(RFmodel, X_test_rf, y_test_rf)
    
    print(f'Linear Regression MAE: ${round(lr_mae,2)}')
    print(f'Random Forest MAE: ${round(rf_mae,2)}')



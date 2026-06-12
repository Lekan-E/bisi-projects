# import libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# linear regression model
def train_LRmodel(X, y):

    condo = X.property_type_Condo

    # split the data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=condo)

    # tarin linear model
    lrmodel = LinearRegression().fit(X_train, y_train)

    # save the model
    with open('models/LRmodel.pkl', 'wb') as file:
        pickle.dump(lrmodel, file)

    return lrmodel, X_test, y_test


# random forest model
def train_RFmodel(X, y):

    condo = X.property_type_Condo
    
    # split the data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify= condo)

    # tarin linear model
    rfmodel = RandomForestRegressor(n_estimators=200, 
                            criterion='absolute_error').fit(X_train, y_train)

    # save the model
    with open('models/RFmodel.pkl', 'wb') as file:
        pickle.dump(rfmodel, file)

    return rfmodel, X_test, y_test
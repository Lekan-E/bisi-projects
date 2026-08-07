# import libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# linear regression model
def train_RFmodel(X, y):

    # split the data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)

    # scae the data
    scale = MinMaxScaler()
    Xtrain_scaled = scale.fit_transform(X_train)
    Xtest_scaled = scale.transform(X_test)

    # train random forest model
    rfmodel = RandomForestClassifier(n_estimators=200,
                                 max_depth=None,
                                 max_features='sqrt',
                                 random_state=42).fit(Xtrain_scaled, y_train)

    # save the model and the scaler (needed to transform inputs at inference time)
    with open('models/RFmodel.pkl', 'wb') as file:
        pickle.dump(rfmodel, file)

    with open('models/scaler.pkl', 'wb') as file:
        pickle.dump(scale, file)

    return rfmodel, Xtest_scaled, y_test


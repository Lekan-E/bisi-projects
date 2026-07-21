from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_MLPmodel(X, y):

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    # scale the data - fit only on the training set to avoid leaking test data into training
    scale = MinMaxScaler()
    Xtrain_scaled = scale.fit_transform(X_train)
    Xtest_scaled = scale.transform(X_test)

    # train the feed-forward neural network
    mlpmodel = MLPClassifier(hidden_layer_sizes=(32,),
                              activation='relu',
                              max_iter=1000,
                              random_state=123).fit(Xtrain_scaled, y_train)

    # save the model and the scaler
    with open('models/MLPmodel.pkl', 'wb') as file:
        pickle.dump(mlpmodel, file)

    with open('models/scaler.pkl', 'wb') as file:
        pickle.dump(scale, file)

    return mlpmodel, Xtest_scaled, y_test

# import libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# linear regression model
def train_DTmodel(X, y):

    # split the data to training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)

    # scae the data
    scale = MinMaxScaler()
    Xtrain_scaled = scale.fit_transform(X_train)
    Xtest_scaled = scale.transform(X_test)

    # tarin decision tree model
    dtmodel = DecisionTreeClassifier().fit(Xtrain_scaled, y_train)

    # save the model
    with open('models/DTmodel.pkl', 'wb') as file:
        pickle.dump(dtmodel, file)

    return dtmodel, Xtest_scaled, y_test


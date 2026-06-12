import pandas as pd

# split data
def create_var(df):

    df = pd.read_csv('data/raw/final.csv')

    X = df.drop('price', axis = 1)
    y = df['price']

    return X, y
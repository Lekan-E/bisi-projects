import pandas as pd

# split data
def create_dummy_vars(df):

    # create dummy variables
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'], dtype=int)
    df.to_csv('data/processed/Processed_Credit_Dataset.csv')

    # seperate input and target variables
    X = df.drop('Loan_Approved', axis = 1)
    y = df['Loan_Approved']

    return X, y
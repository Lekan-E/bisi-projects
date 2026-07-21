import pandas as pd

def create_dummy_vars(df):

    # create dummy variables for the categorical columns
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'], dtype='int')
    df.to_csv('data/processed/Processed_Admission_Dataset.csv', index=False)

    # separate input and target variables
    X = df.drop('Admit_Chance', axis=1)
    y = df['Admit_Chance']

    return X, y

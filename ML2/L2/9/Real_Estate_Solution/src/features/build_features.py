import pandas as pd

# split data
def create_var(df):

    # feature engineering steps
    # 1. popular
    df['popular'] = ((df.beds == 2) & (df.baths == 2)).astype(int)
    # 2. recession
    df['recession'] = ((df.year_sold >= 2010) & (df.year_sold <= 2013)).astype(int)
    # 3. property age
    df['property_age'] = df.year_sold - df.year_built

    # create dummy varibles fro the 
    df = pd.get_dummies(df, columns=['property_type'], dtype = int, drop_first=True)

    # store processed dataset in data/processed
    df.to_csv('data/processed/Processed_House_Dataset.csv', index = None)

    # seperate input and target variables
    X = df.drop('price', axis = 1)
    y = df['price']

    return X, y
import pandas as pd

def load_and_preprocess_data(file_path):

    # import data from mall_customers.csv
    df = pd.read_csv(file_path)

    # drop the 'Customer_ID' variable, it carries no signal for clustering
    df = df.drop('Customer_ID', axis=1)

    return df

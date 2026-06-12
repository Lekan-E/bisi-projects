import pandas as pd

def load_and__preprocess_data(file_path):

    # import data from final.csv
    df = pd.read_csv(file_path)
    
    # drop the 'price' variables
    df = df.drop('price', axis=1)

    return df
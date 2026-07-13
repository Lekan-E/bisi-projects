import pandas as pd

def load_and_preprocess_data(file_path):

    # import data from final.csv
    df = pd.read_csv(file_path)

    return df
import pandas as pd

def load_and_preprocess_data(file_path):

    # import data from Admission.csv
    df = pd.read_csv(file_path)

    # convert the target variable into a categorical variable: Admit_Chance >= 0.8 -> 1 (admit), else 0
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)

    # drop the 'Serial_No' variable, it carries no signal for prediction
    df = df.drop('Serial_No', axis=1)

    # University_Rating and Research are categorical even though they're stored as numbers
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')

    return df

import pandas as pd

def load_and_preprocess_data(file_path):

    # import data from final.csv
    df = pd.read_csv(file_path)
    
    # impute missing values
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Education'].fillna(df['Education'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['ApplicantIncome'].fillna(df['ApplicantIncome'].median(), inplace=True)
    df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median(), inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)

    df['Loan_Approved'] = df['Loan_Approved'].replace({'Y':1, 'N':0})

    # drop the 'Loan_ID' variables
    df = df.drop('Loan_ID', axis=1)

    return df
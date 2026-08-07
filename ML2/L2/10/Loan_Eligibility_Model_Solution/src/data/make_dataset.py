import pandas as pd

def load_and_preprocess_data(file_path):

    # import data from final.csv
    df = pd.read_csv(file_path)
    
    # impute missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Education'] = df['Education'].fillna(df['Education'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].median())
    df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median())
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Property_Area'] = df['Property_Area'].fillna(df['Property_Area'].mode()[0])

    df['Loan_Approved'] = df['Loan_Approved'].map({'Y': 1, 'N': 0})

    # drop the 'Loan_ID' variables
    df = df.drop('Loan_ID', axis=1)

    return df
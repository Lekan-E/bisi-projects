# Credit Loan Eligibility Prediction Model

Random Forest classifier that predicts whether a loan applicant is eligible for a loan based on
their personal and financial characteristics.

## Project structure

```
Loan_Eligibility_Model_Solution/
├── data/
│   ├── raw/credit.csv                 # original loan application data
│   └── processed/                     # dummy-encoded dataset used for training
├── models/
│   ├── RFmodel.pkl                    # trained Random Forest classifier
│   └── scaler.pkl                     # MinMaxScaler fit on the training split only
├── src/
│   ├── data/make_dataset.py           # load, impute missing values, clean the raw data
│   ├── features/build_features.py     # one-hot encode categorical columns
│   ├── models/train_model.py          # train/test split, scaling, model training
│   ├── models/predict_model.py        # accuracy + confusion matrix
│   └── visualization/visualize.py     # feature importance and confusion matrix plots
├── logger.py                          # Shared logging configuration (writes to logs/app.log)
├── main.py                            # training pipeline entry point
├── streamlit.py                       # interactive eligibility-check app
└── requirements.txt
```

## Model

Missing values in the raw data (gender, marital status, dependents, education, employment,
income, loan amount/term, credit history, property area) are imputed with the mode or median of
each column. Categorical columns are one-hot encoded, and all features are scaled with
`MinMaxScaler` fit only on the training split before training a `RandomForestClassifier`
(`n_estimators=200`).

The Streamlit app collects an applicant's profile, applies the same encoding/scaling used at
training time, and predicts loan eligibility, alongside the model's overall feature importance
chart.

## Logging & error handling

- All model loads and predictions are logged to `logs/app.log` via `logger.py`.
- If the model or scaler fails to load, the app shows a friendly error and stops rather than
  crashing.
- If a prediction fails for a given input, the error is logged and a friendly message is shown
  without taking down the app.


# Predicting Chances of Admission at UCLA

Neural network (MLPClassifier) that predicts whether a student is likely to be admitted to UCLA's Master's program.

## Project structure

```
admission_chances_ucla/
├── data/
│   ├── raw/                     # original Admission.csv
│   └── processed/               # dummy-encoded dataset used for training
├── models/
│   ├── MLPmodel.pkl             # trained neural network
│   └── scaler.pkl               # MinMaxScaler fit on the training split only
├── src/
│   ├── data/make_dataset.py         # load, binarize the target, clean the raw data
│   ├── features/build_features.py   # one-hot encode categorical columns
│   ├── models/train_model.py        # train/test split, scaling, model training
│   └── models/predict_model.py      # accuracy + confusion matrix
├── main.py                      # training pipeline entry point
└── streamlit.py                 # interactive prediction app
```


## Model

The target `Admit_Chance` (0-1) is binarized at a 0.8 threshold, per the notebook, then
predicted as a classification task with an `MLPClassifier`.

This pipeline trains `hidden_layer_sizes=(32,)` with `max_iter=1000`, which reaches **91% test accuracy** — the smallest single-hidden-layer configuration found that clears the target.

Categorical columns (`University_Rating`, `Research`) are one-hot encoded, and all features are scaled with `MinMaxScaler` fit only on the training split before being saved alongside the model.

The Streamlit app collects an applicant's profile, predicts admit/no-admit along with an estimated admission probability, and returns back the submitted profile for confirmation.

## Logging & error handling

- Model/scaler loads and predictions are logged to `logs/app.log` via `logger.py`.
- If the model or scaler fails to load, the app shows a friendly error and stops rather than crashing.
- If a prediction fails for a given input, the error is logged and a friendly message is shown without taking down the app.


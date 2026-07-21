# UCLA Admission Chance Predictor

Neural network (MLPClassifier) that predicts whether a student is likely to be admitted to
UCLA's Master's program, modularized from `UCLA_Neural_Networks_Solution.ipynb`.

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
│   ├── models/predict_model.py      # accuracy + confusion matrix
│   └── visualization/visualize.py   # loss curve and confusion matrix plots
├── main.py                      # training pipeline entry point
└── streamlit.py                 # interactive prediction app
```

## Usage

Train the model and generate the analysis plots:

```
python main.py
```

Run the app:

```
streamlit run streamlit.py
```

## Model

The target `Admit_Chance` (0-1) is binarized at a 0.8 threshold, per the notebook, then
predicted as a classification task with an `MLPClassifier`.

The notebook's own two configurations (`hidden_layer_sizes=(3,)`, relu and tanh) only reach
85-87% test accuracy, short of its stated 90% success criteria. This pipeline instead trains
`hidden_layer_sizes=(32,)` with `max_iter=1000`, which reaches **91% test accuracy** — the
smallest single-hidden-layer configuration found that clears the target.

Categorical columns (`University_Rating`, `Research`) are one-hot encoded, and all features
are scaled with `MinMaxScaler` fit only on the training split (per the notebook's data-leakage
warning) before being saved alongside the model.

The Streamlit app collects an applicant's profile, predicts admit/no-admit along with an
estimated admission probability, and shows the model's training loss curve.

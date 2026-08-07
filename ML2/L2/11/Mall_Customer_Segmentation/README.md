# Mall Customer Segmentation Modle

Unsupervised K-Means clustering model that segments mall customers by annual income and spending score.

## Project structure

```
Mall_Customer_Segmentation/
├── data/
│   ├── raw/                     # original mall_customers.csv
│   └── processed/               # cleaned dataset used for training
├── models/
│   └── KMeansmodel.pkl          # trained clustering model
├── src/
│   ├── data/make_dataset.py         # load + clean the raw data
│   ├── features/build_features.py   # select the clustering features
│   ├── models/train_model.py        # elbow/silhouette analysis + final model training
│   ├── models/predict_model.py      # cluster assignment helpers
│   └── visualization/visualize.py   # elbow, silhouette, and cluster plots
├── main.py                      # training pipeline entry point
└── streamlit.py                 # interactive segmentation app
```


## Model

K-Means trained on `Age`, `Annual_Income`, and `Spending_Score`. Features are standardized
(`StandardScaler`) before clustering since they live on different raw scales. `main.py`
picks the number of clusters by maximizing the Silhouette score across k=3-8; the Silhouette
curve peaks cleanly at k=6, matching the notebook's own conclusion for this feature set.

The Streamlit app assigns a submitted customer to a segment and redraws the cluster plot
(projected onto Income vs. Spending) with that customer highlighted, so the chart reflects
the input rather than showing a static image.

## Logging & error handling

- Model/scaler/data loads and predictions are logged to `logs/app.log` via `logger.py`.
- If the model, scaler, or customer data fails to load, the app shows a friendly error and
  stops rather than crashing.
- If assigning a segment fails for a given input, the error is logged and a friendly message
  is shown without taking down the app.



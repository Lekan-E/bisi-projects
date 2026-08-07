# House Price Prediction

A Streamlit app that estimates the transaction price of a property from its features, using
a Random Forest Regressor trained on historical home sales data. Built for a prospective buyer or
seller to get an instant, data-driven price estimate.

## Project structure

```
Real_Estate_Solution/
├── data/
│   ├── raw/house.csv                  # Raw training data
│   └── processed/                     # Feature-engineered data, written by the pipeline
├── models/
│   ├── LRmodel.pkl                    # Linear Regression baseline
│   └── RFmodel.pkl                    # Random Forest model (the one served by the app)
├── src/
│   ├── data/make_dataset.py           # Load raw data
│   ├── features/build_features.py     # Feature engineering (popular/recession/property_age, dummies)
│   ├── models/train_model.py          # Train LR & RF models, persist to models/
│   ├── models/predict_model.py        # Evaluate a trained model (MAE)
│   └── visualization/visualize.py     # Feature importance chart
├── logger.py                          # Shared logging configuration (writes to logs/app.log)
├── main.py                            # Training pipeline entry point
├── streamlit.py                       # The user-facing app
└── requirements.txt
```


## How it works

The app collects a property's details (year built/sold, type, beds, baths, square footage, lot
size, taxes, insurance) through a form, processes the same features used at training time
(`popular`, `recession`, `property_age`, property-type dummy), and feeds them to the pretrained
Random Forest model to produce an estimated price.

## Logging & error handling

- Model loads and predictions are logged to `logs/app.log` via `logger.py`.
- If the model file fails to load, the app shows a friendly error and stops rather than crashing.
- If a prediction fails for a given input, the error is logged and a friendly message is shown
  without taking down the app.



from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_var
from src.models.train_model import train_LRmodel, train_RFmodel
from src.models.predict_model import evaluate_model
from src.visualization.visualize import plot_feature_importance

from logger import get_logger
logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline.")

        # Load and preprocess the data
        data_path = "data/raw/house.csv"
        df = load_and_preprocess_data(data_path)
        logger.info("Loaded and preprocessed %d rows from %s.", len(df), data_path)

        # Engineer features and separate input/target variables
        X, y = create_var(df)

        # Train both candidate models
        LRmodel, X_test_lr, y_test_lr = train_LRmodel(X, y)
        RFmodel, X_test_rf, y_test_rf = train_RFmodel(X, y)
        logger.info("Model training complete.")

        # Evaluate both models
        lr_mae = evaluate_model(LRmodel, X_test_lr, y_test_lr)
        rf_mae = evaluate_model(RFmodel, X_test_rf, y_test_rf)
        logger.info("Linear Regression MAE: %.2f", lr_mae)
        logger.info("Random Forest MAE: %.2f", rf_mae)

        print(f"Linear Regression MAE: ${round(lr_mae, 2)}")
        print(f"Random Forest MAE: ${round(rf_mae, 2)}")

        # The Random Forest model is the one served by the Streamlit app,
        # so its feature importances are what's worth visualizing
        plot_feature_importance(RFmodel, X)
    except Exception:
        logger.exception("Training pipeline failed.")
        raise

from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_MLPmodel
from src.models.predict_model import evaluate_model
from src.visualization.visualize import plot_loss_curve, plot_confusion_matrix

from logger import get_logger
logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline.")

        # Load and preprocess the data
        data_path = "data/raw/Admission.csv"
        df = load_and_preprocess_data(data_path)
        logger.info("Loaded and preprocessed %d rows from %s.", len(df), data_path)

        # Create dummy variables and separate features and target
        X, y = create_dummy_vars(df)

        # Train the neural network model
        model, X_test_scaled, y_test = train_MLPmodel(X, y)
        logger.info("Model training complete.")

        # Evaluate the model
        accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
        logger.info("Accuracy: %s", accuracy)
        logger.info("Confusion Matrix:\n%s", confusion_mat)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion_mat}")

        plot_loss_curve(model)
        plot_confusion_matrix(y_test, model.predict(X_test_scaled), classes=['Not Admitted', 'Admitted'])
    except Exception:
        logger.exception("Training pipeline failed.")
        raise

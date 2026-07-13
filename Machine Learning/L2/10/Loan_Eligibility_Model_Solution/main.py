from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_DTmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/credit.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the logistic regression model
    model, X_test_scaled, y_test = train_DTmodel(X, y)

    # Evaluate the model
    plot_feature_importance(model, X)
    accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")



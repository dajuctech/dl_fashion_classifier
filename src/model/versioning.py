import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model

def log_model_to_mlflow(model_path):
    mlflow.set_experiment("Fashion_MNIST_Classification")
    with mlflow.start_run():
        model = load_model(model_path)
        mlflow.keras.log_model(model, "model")
        print(f"Model {model_path} logged to MLflow.")

if __name__ == "__main__":
    model_file = "models/fashion_cnn_best_tuned.h5"  # Example
    log_model_to_mlflow(model_file)

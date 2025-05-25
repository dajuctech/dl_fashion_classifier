import datetime
import os
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from src.model.cnn_model import build_cnn_model, get_callbacks

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., None], x_test[..., None]
    return (x_train, y_train), (x_test, y_test)

def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_cnn_model()
    callbacks = get_callbacks()
    model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=callbacks)

    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"models/fashion_cnn_{timestamp}.h5"
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()

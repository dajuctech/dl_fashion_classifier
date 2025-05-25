import pytest
from tensorflow.keras.models import load_model
import numpy as np

@pytest.fixture
def model():
    model_files = sorted(list(Path("models").glob("*.h5")))
    if not model_files:
        pytest.fail("No model found for testing.")
    return load_model(model_files[-1])

def test_model_accuracy(model):
    # Load test data
    from tensorflow.keras.datasets import fashion_mnist
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test[..., None] / 255.0
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    assert acc > 0.7, f"Model accuracy too low: {acc}"

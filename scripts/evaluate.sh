#!/bin/bash
# Evaluate the trained model
echo "Evaluating the model..."
python3 -c "
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load data
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test[..., None] / 255.0

# Load the latest model
import os, glob
model_files = sorted(glob.glob('models/*.h5'), key=os.path.getmtime)
model_path = model_files[-1] if model_files else None
if model_path:
    model = load_model(model_path)
    loss, acc = model.evaluate(x_test, y_test)
    print(f'Evaluation Results: Loss={loss}, Accuracy={acc}')
else:
    print('No model found in models/ folder')
"

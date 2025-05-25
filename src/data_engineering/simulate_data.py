import os
import pandas as pd
import numpy as np
from PIL import Image

def create_fake_csv(num_samples=1000, output_dir='data/raw'):
    os.makedirs(output_dir, exist_ok=True)
    data = {
        'id': [f'image_{i}' for i in range(num_samples)],
        'label': np.random.randint(0, 10, size=num_samples),
        'feature1': np.random.randn(num_samples),
        'feature2': np.random.randn(num_samples)
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'fake_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Fake CSV created at {csv_path}")

def create_fake_images(num_samples=100, output_dir='data/raw/images'):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        img_array = np.random.randint(0, 255, size=(28,28), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f'image_{i}.png'))
    print(f"Fake images created in {output_dir}")

if __name__ == "__main__":
    create_fake_csv()
    create_fake_images()

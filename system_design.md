# 🧪 System Design for `dl_fashion_classifier`

## 📥 1️⃣ Data Ingestion
- Simulate raw data ingestion from `/data/raw/` (CSV and images).
- Load data using Pandas and PIL (image processing).

## 🛠️ 2️⃣ Preprocessing
- Clean data (drop NA, validate images).
- Normalize pixel values to [0, 1].
- Convert images to tensors with shape `(batch_size, 28, 28, 1)`.

## 🧠 3️⃣ Model Training
- Build a CNN using Keras with:
  - Conv2D, MaxPooling2D, BatchNormalization, Dense layers.
  - Callbacks: EarlyStopping, ReduceLROnPlateau.
- Train on Fashion MNIST or simulated data.

## 🌐 4️⃣ API Layer
- Expose a REST API (Flask/FastAPI) with `/predict` endpoint.
- Accept image input, return class prediction.
- Future: Add authentication and rate limiting.

## ☁️ 5️⃣ Cloud Integration
- Save models to AWS S3 or mock cloud storage.
- Use MLflow to track experiments and models.

## 🔎 6️⃣ Business Intelligence
- Generate reports: accuracy, loss curves, confusion matrix.
- Visualize data and performance using matplotlib/plotly.

## 📦 7️⃣ Deployment
- Containerize with Docker and deploy to a cloud service (AWS/GCP).

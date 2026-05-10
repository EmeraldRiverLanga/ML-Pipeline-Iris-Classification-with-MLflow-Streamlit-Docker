# ML Pipeline: Iris Classification with MLflow, Streamlit & Docker

## Overview

A full end-to-end machine learning pipeline built on the Iris dataset — from model training and experiment tracking to a deployable web application.

## Technologies Used

- **Python** — core language
- **Keras / TensorFlow** — neural network model
- **MLflow** — experiment tracking and model versioning
- **Streamlit** — interactive web interface
- **Docker / Docker Compose** — containerisation and deployment
- **scikit-learn** — data preprocessing and metrics
- **pandas / numpy** — data handling
- **joblib** — saving scaler and label binarizer

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally

```bash
# Step 1 — train the model
python train.py

# Step 2 — view MLflow experiments
mlflow ui
# Open http://localhost:5000

# Step 3 — run Streamlit app
streamlit run app.py
# Open http://localhost:8501
```

### Run with Docker

```bash
docker compose up --build
```

- Streamlit app: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

## Project Structure

```
project/
├── train.py              # Model training and MLflow logging
├── app.py                # Streamlit application
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Multi-service setup (Streamlit + MLflow)
├── requirements.txt      # Python dependencies
├── .dockerignore
├── best_model.keras      # Saved best model (primary)
├── best_model.h5         # Saved best model (legacy format)
├── scaler.pkl            # Fitted StandardScaler
├── label_binarizer.pkl   # Fitted LabelBinarizer
└── README.md
```

## Experiments

| Experiment | Units (L1/L2) | Dropout | Batch size | F1-score |
|---|---|---|---|---|
| experiment_small | 32 / 16 | 0.2 | 16 | 1.0 |
| experiment_medium | 64 / 32 | 0.3 | 16 | 1.0 |
| experiment_large | 128 / 64 | 0.4 | 8 | 1.0 |

The best model is selected automatically based on F1-score and saved to `best_model.keras` (also exported as `best_model.h5` for legacy compatibility).

## Model Details

- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Architecture:** Dense neural network with 2 hidden layers, Dropout regularisation, EarlyStopping
- **Optimiser:** Adam
- **Loss:** Categorical crossentropy
- **Evaluation:** Accuracy, F1-score (weighted)

## Key Concepts Demonstrated

- **Experiment tracking** — logging hyperparameters and metrics with MLflow
- **Model selection** — automatic best model selection based on F1-score
- **Web deployment** — serving predictions through a Streamlit interface
- **Containerisation** — packaging the full system with Docker Compose
- **ML pipeline** — end-to-end flow from training to deployable application

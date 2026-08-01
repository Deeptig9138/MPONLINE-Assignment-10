# Heart Disease Prediction REST API & Cloud Deployment

**Author:** Deepti Gupta

**Registration Number:** 23MEI10007

**Application Number:** IN26011816

**Batch Number:** 1A

**Email ID:** deepti.23mei10007@vitbhopal.ac.in

**Render Deployment URL:** [Watch it here!]()

---

## Objective

The goal of this project is to build an end-to-end Machine Learning pipeline that predicts heart disease risk based on clinical parameters and deploys a production-ready REST API using Flask, Gunicorn, GitHub, and Render.

## Dataset

- **Name:** Heart Disease Dataset
- **Kaggle Link:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## Project Repository Structure

```
Heart-Disease-Deployment/
├── heart.csv
├── train_model.py
├── app.py
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies & Libraries Used

- **Language:** Python 3.10
- **Machine Learning & Data Processing:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Web Framework & WSGI Server:** `Flask`, `gunicorn`
- **Version Control & Hosting:** Git, GitHub, Render

---

## Methodology

1. **Data Preprocessing & Analysis:** Loaded `heart.csv`, evaluated numerical clinical features, verified zero missing values, and partitioned the data into 80% training and 20% testing sets.
2. **Model Training & Serialization:** Trained a `RandomForestClassifier` on clinical indicators, evaluated prediction accuracy, and serialized the trained model as `model.pkl` using `joblib`.
3. **API Development:** Created a Flask REST API exposing a `/predict` `POST` endpoint accepting patient parameters as JSON input and returning binary prediction labels.
4. **Cloud Deployment:** Hosted the service publicly on Render using the Gunicorn WSGI web server, backed by GitHub continuous deployment.

---

## REST API Usage & Examples

### Root Health Check Endpoint

- **URL:** `https://heart-disease-rf6i.onrender.com/`

**Response:**

```json
{
  "message": "Heart Disease Prediction REST API is running successfully.",
  "status": "API active"
}
```

---

### Prediction Endpoint

- **URL:** ``
- **Headers:** `Content-Type: application/json`

#### Example Input Payload

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

#### Example Output Response

```json
{
  "prediction": "No Heart Disease Detected",
  "prediction_code": 0
}
```

---

## Conclusion

In this assignment, a Machine Learning pipeline was successfully developed and deployed to predict heart disease risk based on clinical patient parameters. A Random Forest Classifier model was trained on the Heart Disease dataset, achieving optimal accuracy, and serialized using `joblib`. The model was integrated into a Flask REST API and deployed as a live web service on Render using Gunicorn. During deployment, challenges such as local port conflicts (macOS AirPlay binding port 5000) and build errors caused by default Python version mismatches (Python 3.14 on Render vs. Python 3.10 locally) were encountered and resolved by explicitly pinning environment variables. This project highlights the critical role of MLOps in production engineering. By leveraging version control with GitHub, environment configuration, dependency management, and automated continuous deployment platforms, MLOps bridges the gap between isolated machine learning models and scalable, accessible API services in real-world applications.

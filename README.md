# 💳 Financial Fraud Detection and Deployment

## 📌 Project Overview

An end-to-end Financial Fraud Detection system built using Machine Learning, FastAPI, Streamlit, and Docker.

The system predicts whether a financial transaction is legitimate or potentially fraudulent using a trained Random Forest model.

## 🎯 Objectives

- Detect potentially fraudulent financial transactions.
- Build a REST API using FastAPI.
- Create an interactive frontend using Streamlit.
- Package and deploy the application using Docker.

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- FastAPI
- Streamlit
- Joblib
- Docker
- Git and GitHub

## 🧠 Machine Learning Model

**Model:** Random Forest Classifier

The model uses transaction features such as Time, V1–V28, and Amount.

Prediction classes:
- `0` – Legitimate transaction
- `1` – Fraudulent transaction

## 📁 Project Structure

```text
financial-fraud-detection-deployment/
│
├── Project_1_Results/
├── app/
├── model/
├── Dockerfile
├── fraud_detection_pipeline.joblib
├── frontend.py
├── main.py
└── requirements.txt
```

## 🚀 Run the Project Using Docker

### 1. Build the Docker image

```bash
docker build -t fraud-detection-api .
```

### 2. Run the Docker container

```bash
docker run -d -p 8000:8000 -p 8501:8501 --name fraud-detection-container fraud-detection-api
```

### 3. Access the application

- Frontend: http://localhost:8501
- API documentation: http://localhost:8000/docs
- API health check: http://localhost:8000/health

## 🔍 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API status |
| POST | `/predict` | Predict transaction status |

## 👨‍💻 Author

**Sudhan**

B.Tech – Artificial Intelligence and Data Science

## 📌 Project Purpose

This project demonstrates the complete machine learning deployment workflow, from model packaging and API development to frontend integration and Docker containerization.

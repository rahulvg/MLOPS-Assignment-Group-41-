Heart Disease Prediction System – MLOps Report

Course: MLOps
Assignment: End-to-End MLOps Pipeline
Dataset: UCI Heart Disease Dataset
Group: Group 41
Repository: https://github.com/rahulvg/MLOPS-Assignment-Group-41-

1. Problem Statement

The objective of this project is to design, develop, and deploy a scalable and reproducible machine learning system to predict the risk of heart disease based on patient health attributes. The solution follows modern MLOps best practices including experiment tracking, CI/CD automation, containerization, Kubernetes deployment, and monitoring.

2. Setup and Installation Instructions
2.1 Local Environment

Python Version
Python 3.10

Install Dependencies

pip install -r requirements.txt


Launch MLflow UI

mlflow ui --backend-store-uri sqlite:///mlflow.db


Access:

http://localhost:5000

2.2 Kubernetes (Local Deployment)

Start Minikube with containerd runtime:

minikube start --container-runtime=containerd


Build Docker image inside Minikube:

minikube image build -t heart-disease-api .


Deploy application:

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml


Expose service:

minikube service heart-disease-service

3. Data Acquisition and Exploratory Data Analysis
3.1 Dataset

Source: UCI Machine Learning Repository

Format: CSV

Target: Binary classification (presence or absence of heart disease)

3.2 Preprocessing

Missing values handled

Numerical features scaled using StandardScaler

Target variable encoded

Preprocessing implemented using a pipeline for reproducibility

3.3 Exploratory Data Analysis

Feature distributions analyzed using histograms

Correlation heatmap used to identify relationships between features

Class balance verified

4. Feature Engineering and Model Development
4.1 Feature Pipeline

A unified scikit-learn Pipeline was used:

StandardScaler

Classifier

This ensures identical preprocessing during training and inference.

4.2 Models Trained
Model	Hyperparameters
Logistic Regression	C ∈ {0.1, 1.0, 10.0}
Random Forest	n_estimators ∈ {100, 200}, max_depth ∈ {None, 10}
4.3 Evaluation Strategy

5-fold cross-validation

Metrics used:

Accuracy

Precision

Recall

ROC-AUC

4.4 Model Selection

Selected Model: Logistic Regression (C = 0.1)

Reasoning

Best balance between accuracy and recall

Lower variance across folds

Simpler and more interpretable

Stable and reliable for deployment

5. Experiment Tracking

MLflow was integrated to track:

Model parameters

Cross-validation metrics

Model artifacts

Experiments are logged under a dedicated MLflow experiment, enabling easy comparison across different model configurations.

6. Model Packaging and Reproducibility

Final model saved as a serialized scikit-learn Pipeline

Preprocessing included within the model

Reproducible inference guaranteed

Dependencies listed in requirements.txt

Artifacts stored via MLflow

7. CI/CD Pipeline
7.1 Tools

GitHub Actions

Pytest

Flake8

Docker

7.2 Pipeline Stages

Code linting

Unit testing

Model training

Docker image build

API smoke test

Each workflow run logs execution details and artifacts.

8. Containerization and Deployment
8.1 Dockerized API

FastAPI-based service

/predict endpoint

Accepts JSON input

Returns prediction and confidence score

8.2 Kubernetes Deployment

Local Kubernetes using Minikube

Deployment and NodePort Service manifests

API successfully exposed and verified using curl and Postman

9. Monitoring and Logging
9.1 Logging

Request-level logging implemented using FastAPI middleware

Logs include endpoint, HTTP status, and latency

Logs collected via Kubernetes pod logs

9.2 Monitoring

Prometheus-compatible /metrics endpoint exposed

Metrics include request count and latency

Enables integration with Prometheus and Grafana

10. Architecture Overview

High-level system architecture:

Client (Postman / curl)
→ FastAPI API (/predict, /metrics)
→ Scikit-learn Pipeline
→ Kubernetes Pod
→ NodePort Service

CI/CD and experiment tracking are handled using GitHub Actions and MLflow respectively.

11. Code Repository

https://github.com/rahulvg/MLOPS-Assignment-Group-41-

12. Conclusion

This project demonstrates a complete, production-style MLOps workflow covering data analysis, model development, experiment tracking, CI/CD automation, containerization, Kubernetes deployment, and monitoring. The system is reproducible, scalable, and aligned with real-world MLOps practices.

Launch mlflow with local db file : mlflow ui --backend-store-uri sqlite:///E:/RGI3/MLOPS/mlflow.db


minikube image build -t heart-disease-api .
kubectl delete deployment heart-disease-api
kubectl apply -f k8s/deployment.yaml

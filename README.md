🍷 WineMetric

Model-Assisted Wine Quality Prediction

<p align="center">
  <strong>An end-to-end machine learning pipeline for predicting red and white wine quality from physicochemical measurements.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?logo=githubactions&logoColor=white" alt="GitHub Actions">
</p>

From lab sheet to quality score — before the bottle is filled.

📌 Project Overview

WineMetric is an end-to-end machine learning project that predicts red and white wine quality using physicochemical laboratory measurements and wine type.

The project demonstrates a production-oriented ML workflow covering:

Data Ingestion → Validation → Transformation → Model Training → Model Comparison → Evaluation → MLflow Tracking → Flask Prediction App → Docker → GitHub Actions CI

Wine quality is traditionally evaluated through sensory assessment. WineMetric explores how laboratory measurements can support an earlier, data-driven estimate of wine quality while recognizing that machine learning does not replace professional tasting.

✨ Project Highlights

Area

Implementation

Problem Type

Regression

Production Model

ExtraTreesRegressor

Wine Types

Red + White

Model Inputs

12

Training Rows

43,700

Test Rows

1,300

Experiment Tracking

MLflow

Web Application

Flask

Testing

pytest

Containerization

Docker

CI

GitHub Actions

📊 Production Model Performance

ExtraTreesRegressor

Metric

Score

RMSE

0.5974

MAE

0.3848

R²

0.5315

The final metrics are calculated on the same 1,300-row test set used for model comparison.

Model Comparison

Model

RMSE ↓

MAE ↓

R² ↑

Extra Trees

0.5974

0.3848

0.5315

Random Forest

0.6431

0.4405

0.4571

Gradient Boosting

0.8148

0.6311

0.1287

ElasticNet

0.8230

0.6417

0.1109

Extra Trees achieved the strongest overall performance and was selected as the production model.

🧠 Machine Learning Workflow

flowchart TD
    A[Wine Quality Dataset] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    E --> G[Model Comparison]
    F --> H[MLflow Tracking]
    G --> H
    G --> I[Extra Trees Selected]
    I --> J[model.joblib]
    J --> K[Prediction Pipeline]
    K --> L[Flask Web App]
    L --> M[Gunicorn]
    M --> N[Docker Container]
    O[pytest] --> P[GitHub Actions CI]
    N --> P

🗂️ Dataset

WineMetric uses a prepared red and white wine quality dataset based on the UCI Wine Quality data.

Dataset Characteristics

Item

Value

Total working rows

45,000

Training rows

43,700

Test rows

1,300

Model input features

12

Target variables

1

Input Features

#

Feature

1

Fixed acidity

2

Volatile acidity

3

Citric acid

4

Residual sugar

5

Chlorides

6

Free sulfur dioxide

7

Total sulfur dioxide

8

Density

9

pH

10

Sulphates

11

Alcohol

12

Wine type (red or white)

Target: quality

⚙️ Pipeline Components

1. Data Ingestion

Downloads and extracts the prepared WineMetric dataset package.

artifacts/data_ingestion/
├── data.zip
└── winemetric_quality/
    ├── winemetric_quality.csv
    ├── winemetric_train.csv
    └── winemetric_test.csv

2. Data Validation

Validates the incoming dataset against the expected schema defined in:

schema.yaml

Validation status is written to:

artifacts/data_validation/status.txt

The pipeline stops if the required schema is not satisfied.

3. Data Transformation

Prepares the fixed training and test files used by the model pipeline.

Training samples: 43,700
Testing samples:   1,300

Generated files:

artifacts/data_transformation/train.csv
artifacts/data_transformation/test.csv

4. Model Training

WineMetric originally used ElasticNet Regression as a baseline.

After model comparison, Extra Trees Regression achieved the strongest results and was selected as the production model.

The trained model is stored at:

artifacts/model_trainer/model.joblib

Current model configuration:

ExtraTrees:
  n_estimators: 200
  random_state: 42
  n_jobs: -1

🔬 MLflow Experiment Tracking

WineMetric integrates MLflow to track machine learning experiments.

Tracked information includes:

model parameters

RMSE

MAE

R²

trained model artifacts

Experiments:

WineMetric
WineMetric-Model-Comparison

Start MLflow locally:

mlflow server --host 127.0.0.1 --port 5000 --workers 1

Open:

http://127.0.0.1:5000

🔮 Prediction Pipeline

The prediction pipeline validates incoming data before running inference.

Validation includes:

missing features

unexpected features

non-numeric chemistry values

valid wine type (red or white)

correct feature ordering

Example:

from mlProject.pipeline.prediction import PredictionPipeline

features = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": "red",
}

pipeline = PredictionPipeline()
prediction = pipeline.predict(features)

print(prediction)

🌐 Flask Web Application

WineMetric includes a browser-based prediction interface built with Flask.

Users can:

Select red or white wine

Enter 11 physicochemical measurements

Submit the form

Receive a predicted wine quality score

The interface also supports wine-type-specific validation ranges and quick demo samples.

Run locally:

python app.py

Open:

http://127.0.0.1:8080

✅ Automated Testing

WineMetric uses pytest for automated validation.

Prediction Tests

valid prediction

missing feature validation

unexpected feature validation

non-numeric input validation

Flask Tests

home page response

valid prediction request

invalid prediction handling

Run:

pytest -v

Current result:

7 passed

🔁 Continuous Integration

WineMetric uses GitHub Actions for continuous integration.

The CI workflow runs for:

pushes to main

pull requests targeting main

CI Pipeline

Checkout Repository
        ↓
Set Up Python 3.12
        ↓
Install Dependencies
        ↓
Install WineMetric
        ↓
Verify Package Imports
        ↓
Run Training Pipeline
        ↓
Run pytest
        ↓
Build Docker Image
        ↓
CI Passed

Workflow:

.github/workflows/ci.yml

🐳 Docker

Build

docker build -t winemetric:1.0 .

Run

docker run --rm --name winemetric-app -p 8080:8080 winemetric:1.0

Open:

http://localhost:8080

The container uses:

Python 3.12

Flask

Gunicorn

non-root application user

Extra Trees production model

▶️ Run the Complete Pipeline

python main.py

Pipeline sequence:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
MLflow Tracking

Expected successful completion:

WineMetric pipeline completed successfully

📁 Project Structure

End-to-end-ML-Project-with-MLFlow/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── config/
│   └── config.yaml
│
├── src/
│   └── mlProject/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── pipeline/
│       └── utils/
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── test_flask_app.py
│   └── test_prediction.py
│
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── main.py
├── params.yaml
├── pytest.ini
├── requirements.txt
├── schema.yaml
└── setup.py

🧰 Technology Stack

Category

Technologies

Machine Learning

Python, scikit-learn, Extra Trees Regression, MLflow

Data Processing

Pandas, NumPy

Web

Flask, Gunicorn, HTML, CSS, JavaScript

Configuration

PyYAML, python-box

Serialization

Joblib

Testing

pytest

DevOps

Docker, GitHub Actions, Git, GitHub

♻️ Reproducibility

A clean environment can:

Clone the repository

Install dependencies

Download the prepared dataset package

Validate the dataset

Prepare the training and test files

Train the model

Evaluate the model

Generate the model artifact

Run automated tests

Build the Docker image

The trained model does not need to be manually committed to Git.

🔐 Security Considerations

WineMetric follows several secure-development practices:

environment files are excluded from Git

generated artifacts are excluded from version control

prediction inputs are validated before inference

Docker runs the application as a non-root user

Flask debug mode is not used as the production container server

Gunicorn serves the production container

GitHub Actions uses read-only repository content permission

secrets should be supplied through environment variables or GitHub Secrets

⚠️ Limitations

WineMetric predicts quality from physicochemical measurements and wine type only.

The model:

does not replace professional sensory evaluation

depends on the quality and representativeness of the training data

should be interpreted cautiously for rare quality values with limited examples

should not be treated as a guarantee of consumer preference or commercial wine quality

🚀 Future Improvements

Potential future extensions include:

hyperparameter tuning

broader model comparison

feature importance visualization

model registry integration

cloud deployment

model monitoring

prediction drift monitoring

REST API endpoint

additional integration tests

📚 Reference

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
Modeling wine preferences by data mining from physicochemical properties.
Decision Support Systems, 47(4), 547–553.

Dataset: UCI Machine Learning Repository — Wine Quality Dataset

🎓 Capstone

SAIT — Data Analytics
Tech CapCon Spring 2026

Team Members

Sharndeep Kaur

Sajid Bapu

Harpreet Kaur

Rakhsha Varu

Supervisor

Tee Wijesooriya

📄 License

This project was developed for academic and educational purposes as part of the SAIT Data Analytics Capstone Project.
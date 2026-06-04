# Student Exam Performance Predictor

A production-style machine learning project that predicts a student's math score using reading score, writing score, and academic background features. The project includes an end-to-end training pipeline, saved preprocessing/model artifacts, and a Flask web application for live predictions.

## Project Highlights

- Built a complete ML workflow from data ingestion to model serving.
- Compared multiple regression algorithms including Linear Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost, KNN, Decision Tree, and AdaBoost.
- Used scikit-learn pipelines for missing-value handling, scaling, and one-hot encoding.
- Persisted the trained model and preprocessor with `dill` for reusable inference.
- Served predictions through a clean Flask web interface.
- Structured the codebase with reusable components, custom exception handling, logging, and package setup.

## Problem Statement

The goal is to predict a student's `math_score` based on:

- Gender
- Race or ethnicity group
- Parental level of education
- Lunch type
- Test preparation course status
- Reading score
- Writing score

This is a supervised regression problem using the Student Performance dataset.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- CatBoost
- XGBoost
- Flask
- dill
- HTML/CSS

## Project Structure

```text
MLproject/
+-- app.py
+-- requirements.txt
+-- setup.py
+-- README.md
+-- artifacts/
|   +-- model.pkl
|   +-- preprocessor.pkl
|   +-- raw.csv
|   +-- train.csv
|   +-- test.csv
+-- notebook/
|   +-- 1 . EDA STUDENT PERFORMANCE .ipynb
|   +-- 2. MODEL TRAINING.ipynb
|   +-- data/
|       +-- stud.csv
+-- src/
|   +-- components/
|   |   +-- data_ingestion.py
|   |   +-- data_transformation.py
|   |   +-- model_trainer.py
|   +-- pipeline/
|   |   +-- predict_pipeline.py
|   |   +-- train_pipeline.py
|   +-- exception.py
|   +-- logger.py
|   +-- utils.py
+-- templates/
    +-- home.html
    +-- index.html
```

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training pipeline:

```bash
python src/components/data_ingestion.py
```

This command:

1. Loads the dataset from `notebook/data/stud.csv`.
2. Splits it into train and test sets.
3. Builds and saves the preprocessing pipeline.
4. Trains and evaluates multiple regression models.
5. Saves the best performing model to `artifacts/model.pkl`.

## Run the Web App

Start the Flask application:

```bash
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

Fill in the student details and submit the form to get the predicted math score.

## Machine Learning Pipeline

The project follows this workflow:

```text
Raw Data
   |
Train/Test Split
   |
Preprocessing
   +-- Numerical columns: median imputation + standard scaling
   +-- Categorical columns: mode imputation + one-hot encoding + scaling
   |
Model Training and Hyperparameter Search
   |
Best Model Selection
   |
Saved Artifacts
   |
Flask Prediction App
```

## Models Evaluated

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- K-Neighbors Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

The best model is selected using the test R2 score and saved for inference.

## Key Files

- `app.py`: Flask app entry point.
- `src/components/data_ingestion.py`: Loads data and creates train/test files.
- `src/components/data_transformation.py`: Builds the preprocessing pipeline.
- `src/components/model_trainer.py`: Trains and selects the best regression model.
- `src/pipeline/predict_pipeline.py`: Loads saved artifacts and returns predictions.
- `src/utils.py`: Shared utilities for saving/loading objects and model evaluation.

## Verification

The prediction pipeline was tested with a sample input and returned a valid math score prediction. The Flask form field names were also aligned with the inference pipeline so browser submissions pass the correct features to the model.

## Future Improvements

- Add automated tests for the training and prediction pipelines.
- Add model metrics tracking with MLflow or a similar experiment manager.
- Deploy the Flask app on Render, Railway, or AWS.
- Add a Dockerfile for reproducible local and cloud execution.
- Improve the web UI with validation messages and responsive styling.

## Author

**Sidh**

This project demonstrates practical machine learning engineering skills: data processing, model comparison, artifact persistence, web deployment basics, and clean Python project structure.

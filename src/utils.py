import os 
import sys 
import pandas as pd 
import numpy as np  
from src.exception import CustomException
from src.logger import logging  
import dill

def save_object(file_path, obj):    
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Ensure directory exists

    
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)   
    

from sklearn.metrics import r2_score
import sys
from src.exception import CustomException

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Loop through each model in the dictionary
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

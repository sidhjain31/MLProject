import os
import sys
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name} with hyperparameter tuning...")

            param_grid = params.get(model_name, {})

            if len(param_grid) > 0:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_  # already fitted
                models[model_name] = best_model
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

            logging.info(f"{model_name} - Train R2: {train_score:.3f}, Test R2: {test_score:.3f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)

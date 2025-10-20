import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # ✅ Model dictionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # ✅ Simplified parameter grid for tuning
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [50, 100, 200],
                    "subsample": [0.8, 1.0]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9]
                },
                "XGBoost Regressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [50, 100, 200]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [50, 100, 200]
                }
            }

            logging.info("Starting model evaluation with hyperparameter tuning...")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # ✅ Get best model
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² >= 0.6")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"✅ Best model: {best_model_name} | R²: {best_model_score:.4f}")

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            print(f"\n✅ Final R² Score of Best Model ({best_model_name}): {r2:.4f}\n")

            return r2

        except Exception as e:
            raise CustomException(e, sys)

import yaml
import os
import sys
import numpy as np
import pickle

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:   # ✅ FIXED (text mode)
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Saving object...")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File does not exist: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)   # ✅ removed debug print

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            logging.info(f"Training model: {model_name}")

            if params:
                gs = GridSearchCV(model, params, cv=3)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # ✅ FIXED: classification metric
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)
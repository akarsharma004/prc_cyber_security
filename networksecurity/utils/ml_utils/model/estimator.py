import sys
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException


class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        try:
            # Ensure input is numpy array
            if not isinstance(x, (list, np.ndarray)):
                raise Exception("Input must be list or numpy array")

            x = np.array(x)

            # Transform
            x_transformed = self.preprocessor.transform(x)

            # Predict
            y_hat = self.model.predict(x_transformed)

            return y_hat

        except Exception as e:
            raise NetworkSecurityException(e, sys)
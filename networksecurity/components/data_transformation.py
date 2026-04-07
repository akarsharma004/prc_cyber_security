from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object
)

import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(file_path):
                raise Exception(f"File not found: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates preprocessing pipeline (KNN Imputer)
        """
        try:
            logging.info("Creating KNN Imputer pipeline")

            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            processor = Pipeline([
                ("imputer", imputer)
            ])

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation")

        try:
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # ===== TRAIN =====
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            # ===== TEST =====
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # ===== PREPROCESS =====
            preprocessor = self.get_data_transformer_object()

            transformed_input_train = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test = preprocessor.transform(input_feature_test_df)

            # ===== COMBINE =====
            train_arr = np.c_[transformed_input_train, target_feature_train_df.values]
            test_arr = np.c_[transformed_input_test, target_feature_test_df.values]

            # ===== SAVE ARRAYS =====
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            # ===== SAVE PREPROCESSOR =====
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            # OPTIONAL (safe path creation)
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/preprocessor.pkl", preprocessor)

            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
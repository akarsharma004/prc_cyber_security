from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from MongoDB and return as DataFrame
        """
        try:
            if MONGO_DB_URL is None:
                raise Exception("MONGO_DB_URL not found in environment variables")

            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info(f"Connecting to MongoDB: {database_name}.{collection_name}")

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                raise Exception("No data found in MongoDB collection")

            # Drop MongoDB auto-generated column
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            # Replace string "na" with np.nan
            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Data fetched successfully with shape: {df.shape}")

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Save raw data into feature store (CSV)
        """
        try:
            file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(file_path, index=False, header=True)

            logging.info(f"Feature store file saved at: {file_path}")

            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Split dataset into train and test and save them
        """
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info("Train-test split completed")

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            logging.info(f"Train file saved at: {train_path}")
            logging.info(f"Test file saved at: {test_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Complete data ingestion pipeline
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion completed successfully")

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
# from networksecurity.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            self.s3_sync = S3Sync()
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ========================
    # DATA INGESTION
    # ========================
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            config = DataIngestionConfig(self.training_pipeline_config)

            logging.info("Starting Data Ingestion")
            ingestion = DataIngestion(config)
            artifact = ingestion.initiate_data_ingestion()

            logging.info(f"Data Ingestion completed: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ========================
    # DATA VALIDATION
    # ========================
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            config = DataValidationConfig(self.training_pipeline_config)

            logging.info("Starting Data Validation")
            validation = DataValidation(data_ingestion_artifact, config)
            artifact = validation.initiate_data_validation()

            logging.info(f"Data Validation completed: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ========================
    # DATA TRANSFORMATION
    # ========================
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            config = DataTransformationConfig(self.training_pipeline_config)

            logging.info("Starting Data Transformation")
            transformation = DataTransformation(data_validation_artifact, config)
            artifact = transformation.initiate_data_transformation()

            logging.info(f"Data Transformation completed: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ========================
    # MODEL TRAINING
    # ========================
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            config = ModelTrainerConfig(self.training_pipeline_config)

            logging.info("Starting Model Training")
            trainer = ModelTrainer(config, data_transformation_artifact)
            artifact = trainer.initiate_model_trainer()

            logging.info(f"Model Training completed: {artifact}")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # # ========================
    # # S3 SYNC (SAFE)
    # # ========================
    # def sync_artifact_dir_to_s3(self):
    #     try:
    #         aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"

    #         logging.info("Uploading artifacts to S3")
    #         self.s3_sync.sync_folder_to_s3(
    #             folder=self.training_pipeline_config.artifact_dir,
    #             aws_bucket_url=aws_bucket_url
    #         )

    #     except Exception as e:
    #         logging.warning(f"S3 artifact sync failed (ignored): {e}")

    # def sync_saved_model_dir_to_s3(self):
    #     try:
    #         aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"

    #         logging.info("Uploading model to S3")
    #         self.s3_sync.sync_folder_to_s3(
    #             folder=self.training_pipeline_config.model_dir,
    #             aws_bucket_url=aws_bucket_url
    #         )

    #     except Exception as e:
    #         logging.warning(f"S3 model sync failed (ignored): {e}")

    # ========================
    # MAIN PIPELINE
    # ========================
    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            logging.info("Pipeline started")

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(ingestion_artifact)

            # 🔴 HARD STOP IF DATA INVALID
            if not validation_artifact.validation_status:
                raise Exception("Data validation failed. Stopping pipeline.")

            transformation_artifact = self.start_data_transformation(validation_artifact)

            trainer_artifact = self.start_model_trainer(transformation_artifact)

            # Non-blocking sync
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            logging.info("Pipeline completed successfully")

            return trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
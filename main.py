import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline")

        # ===== CONFIG =====
        training_pipeline_config = TrainingPipelineConfig()

        # ===== DATA INGESTION =====
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiating data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
        print(data_ingestion_artifact)

        # ===== DATA VALIDATION =====
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)

        logging.info("Initiating data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data validation completed: {data_validation_artifact}")
        print(data_validation_artifact)

        # 🚨 HARD STOP IF VALIDATION FAILS
        if not data_validation_artifact.validation_status:
            raise Exception("Data validation failed. Stopping pipeline.")

        # ===== DATA TRANSFORMATION =====
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact,
            data_transformation_config
        )

        logging.info("Initiating data transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation completed: {data_transformation_artifact}")
        print(data_transformation_artifact)

        # ===== MODEL TRAINING =====
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )

        logging.info("Initiating model training")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model training completed: {model_trainer_artifact}")

        print(model_trainer_artifact)

        logging.info("Pipeline completed successfully")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
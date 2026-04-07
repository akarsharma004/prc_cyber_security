import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")   # ✅ FIXED ENV NAME

import pymongo
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# ===== MongoDB =====
if mongo_db_url is None:
    raise Exception("MONGO_DB_URL not found in environment")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ===== FastAPI =====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


# ========================
# ROOT
# ========================
@app.get("/", tags=["auth"])
async def index():
    return RedirectResponse(url="/docs")


# ========================
# TRAIN
# ========================
@app.get("/train")
async def train_route():
    try:
        logging.info("Training triggered via API")

        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        return Response("Training completed successfully")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ========================
# PREDICT
# ========================
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # ===== LOAD MODEL =====
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=model
        )

        # ===== PREDICT =====
        predictions = network_model.predict(df)

        df["predicted_column"] = predictions

        # ===== SAVE OUTPUT =====
        os.makedirs("prediction_output", exist_ok=True)   # ✅ FIX
        output_path = "prediction_output/output.csv"
        df.to_csv(output_path, index=False)

        # ===== RETURN HTML =====
        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ========================
# RUN APP
# ========================
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
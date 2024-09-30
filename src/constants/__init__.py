import os,sys
from datetime import datetime

## artifact -> pipeline folder -> timestamp -> output

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%M-%d %H-%M-%S')}"

CURRENT_TIME_STAMP=get_current_time_stamp()

# Data Set path 
ROOT_DIR_KEY=os.getcwd()
DATA_DIR="dataset"
DATA_DIR_KEY="finalTrain.csv"

## creating a folder to store the output
ARTIFACT_DIR_KEY="artifact"
## Data Ingestion related Variable
DATA_INGESTION_KEY='data_ingestion' # this is used t o store raw amd ingested
DATA_INGESTION_RAW_DATA_DIR='raw_data_dir' # this is used to store raw.csv
DATA_INGESTION_INGESTED_DATA_DIR_KEY='ingested_dir' # this is used to store train.csv & test.csv
RAW_DATA_DIR_KEY="raw.csv"
TRAIN_DATA_DIR_KEY="train.csv"
TEST_DATA_DIR_KEY="test.csv"

# Data Transformation related variable

DATA_TRANSFORMATION_ARTIFACT = "data_transformation"
DATA_PREPROCCED_DIR = "procceor"
DATA_TRANSFORMTION_PROCESSING_OBJ = "processor.pkl"
DATA_TRANSFORM_DIR = "transformation"
TRANSFORM_TRAIN_DIR_KEY = "train.csv"
TRANSFORM_TEST_DIR_KEY = "test.csv"


## model trainer

MODEL_TRAINER_KEY="model_trainer"
MODEL_OBJECT='model.pkl'

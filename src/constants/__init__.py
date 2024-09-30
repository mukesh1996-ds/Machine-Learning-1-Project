import os,sys
from datetime import datetime

## artifact -> pipeline folder -> timestamp -> output

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%M-%d %H-%M-%S')}"

CURRENT_TIME_STAPM=get_current_time_stamp()

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


import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import customException
from src.logger import logging

@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            file_path = 'F:\mlprojects\src\notebook\data\stud.csv'
            
            # Check if file exists and is not empty
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file at {file_path} does not exist.")
            
            if os.path.getsize(file_path) == 0:
                raise ValueError("The file is empty.")
            
            # Read the dataset
            df = pd.read_csv(file_path)
            logging.info('Read the dataset as dataframe')
            
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            # Perform train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
        except Exception as e:
            raise customException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

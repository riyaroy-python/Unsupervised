import pandas as pd
import logging

logging.basicConfig(filename='logs/clustering.log', level=logging.INFO)


def load_data(filepath):
    """Load and return the dataset from the given CSV file."""
    try:
        data = pd.read_csv(filepath)
        logging.info("Data loaded successfully from %s", filepath)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise


def preprocess_data(data):
    """Preprocess the data and return numerical features only."""
    try:
        processed = data.drop(columns=['CustomerID', 'Gender'], errors='ignore')
        logging.info("Data preprocessing successful")
        return processed
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise
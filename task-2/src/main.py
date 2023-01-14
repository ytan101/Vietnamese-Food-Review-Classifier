import os

from data_cleaning import DataCleaning
from data_processing import DataProcessing
from model import SentimentClassifier

# import argparse if need to choose between train, test, eval (and also for own input)

if __name__ == "__main__":
    # If clean_data doesn't exist, run data_cleaning
    if not os.path.exists("../data/cleaned/cleaned_data.csv"):
        data_cleaning = DataCleaning()
        data_cleaning.clean_data()

    # Process the data
    data_processing = DataProcessing()
    reviews_dataset = data_processing.process_data()
    
    # Modelling
    model = SentimentClassifier(dataset=reviews_dataset, batch_size=8, truncation_length=512)
    model.training_loop()

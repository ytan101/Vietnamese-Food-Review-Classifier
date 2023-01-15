import os
import argparse

from src.data_cleaning import DataCleaning
from src.data_processing import DataProcessing
from src.model import SentimentClassifierModel


class SentimentClassifier:
    def __init__(self, mode, model_path, text):
        self.model_path = model_path
        self.mode = mode
        self.text = text

        # Process the data
        self.data_processing = DataProcessing()
        reviews_dataset = self.data_processing.process_data()

        # Modelling
        self.model = SentimentClassifierModel(
            dataset=reviews_dataset,
            mode=self.mode,
            batch_size=8,
            truncation_length=512,
            num_epochs=2,
        )

    def train(self):
        self.model.training_loop()
        self.model.testing_loop()

    def infer(self):
        print("printing text")
        processed_text_input = self.data_processing.process_single(self.text)
        print(processed_text_input)
        text_input_dict = {"Review": [processed_text_input]}
        prediction = self.model.inference_loop(self.model_path, text_input_dict)
        print(prediction)
        return prediction


if __name__ == "__main__":
    # If clean_data doesn't exist, run data_cleaning
    if not os.path.exists("../data/cleaned/cleaned_data.csv"):
        data_cleaning = DataCleaning()
        data_cleaning.clean_data()

    parser = argparse.ArgumentParser(
        description="Get the sentiment of Vietnamese food reviews"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="train: enables training loop, infer: predicts your input sentiment",
        default="train",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path, models located in ./data/raw",
        default="../data/models/model_2023-01-15_01-15-00.pt",
    )
    default_text = "Please key in your text"
    parser.add_argument(
        "--text", type=str, help="Text to be analyzed", default=default_text
    )
    args = parser.parse_args()

    sentiment_classifier = SentimentClassifier(
        mode=args.mode, model_path=args.model_path, text=args.text
    )

    if args.mode == "train":
        sentiment_classifier.train()

    if args.mode == "infer":
        sentiment_classifier.infer()

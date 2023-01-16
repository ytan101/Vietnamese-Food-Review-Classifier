import os
import argparse

from src.data_cleaning import DataCleaning
from src.data_processing import DataProcessing
from src.model import SentimentClassifierModel
from rich import print


class SentimentClassifier:
    def __init__(self, mode, model_path, text, current_directory, num_epochs=5):
        self.model_path = model_path
        self.mode = mode
        self.text = text

        # Process the data
        self.data_processing = DataProcessing(current_directory=current_directory)
        reviews_dataset = self.data_processing.process_data()

        # Modelling
        self.model = SentimentClassifierModel(
            dataset=reviews_dataset,
            current_directory=current_directory,
            mode=self.mode,
            batch_size=8,
            truncation_length=512,
            num_epochs=num_epochs,
        )

    def train(self):
        """
        The train function is the main function that runs the training loop. 
        It calls other functions to get data, create a model, load a checkpoint
        if necessary, train for an epoch and report progress. 
        The only parameters passed to this function are those that can be set by flags in our CLI.
        
        """
        self.model.training_loop()
        self.model.testing_loop()

    def infer(self):
        """
        The infer function takes a text string and returns the predicted sentiment.
        The input is processed, tokenized, and fed into the model for inference.
        The output is then returned to the user
        
        :return: A dictionary with the following keys:
        """
        processed_text_input = self.data_processing.process_single(self.text)
        text_input_dict = {"Review": [processed_text_input]}
        prediction = self.model.inference_loop(self.model_path, text_input_dict)
        print(f"[cyan]You're feeling[/cyan] [magenta]{prediction}[/magenta] [cyan]about what you ate[/cyan]")
        return prediction


if __name__ == "__main__":

    current_directory = os.getcwd()

    # If clean_data doesn't exist, run data_cleaning
    if not os.path.exists(f"{current_directory}/data/cleaned/cleaned_data.csv"):
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
        default=f"{current_directory}\data\models\model_2023-01-16_11-11-54.pt",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="number of epochs to train the model for",
        default=5,
    )
    default_text = "Please key in your text"
    parser.add_argument(
        "--text", type=str, help="Text to be analyzed", default=default_text
    )
    args = parser.parse_args()

    sentiment_classifier = SentimentClassifier(
        mode=args.mode, current_directory=current_directory, model_path=args.model_path, text=args.text, num_epochs=args.num_epochs
    )

    if args.mode == "train":
        sentiment_classifier.train()

    if args.mode == "infer":
        sentiment_classifier.infer()

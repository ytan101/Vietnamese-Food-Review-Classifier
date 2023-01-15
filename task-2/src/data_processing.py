import pandas as pd
import os

from rich import print
from datasets import Dataset

class DataProcessing:
    def __init__(self):
        data_file_path = '../data/cleaned/cleaned_data.csv'
        self.reviews_df = pd.read_csv(data_file_path)

    def process_data(self):
        # Apply lowercasing
        self.reviews_df["Review"] = self.reviews_df["Review"].str.lower()

        # Remove stopwords
        self.reviews_df["Review"] = self.reviews_df["Review"].apply(self.remove_vn_stopwords)

        # Convert to Arrow format
        reviews_dataset = Dataset.from_pandas(self.reviews_df)

        # Split into train, val, test
        return self.split_train_val_test(reviews_dataset)

    def process_single(self, text):
        # Apply lowercasing
        text = text.lower()
        text = self.remove_vn_stopwords(text)
        return text
    
    def remove_vn_stopwords(self, text: str):
        vn_stopwords_file = open("../data/vietnamese-stopwords.txt", encoding="utf8")
        vn_stopwords = vn_stopwords_file.read()
        vn_stopwords = vn_stopwords.split("\n")

        for stopword in vn_stopwords:
            text = text.replace(stopword, "")
        return text

    def split_train_val_test(self, dataset):
        # 1st split; train and val
        train_test_dataset = dataset.train_test_split(train_size=0.8, seed=42)

        # 2nd split; train, val, test
        train_val_test_dataset = train_test_dataset["train"].train_test_split(train_size=0.8, seed=42)
        train_val_test_dataset["validation"] = train_val_test_dataset.pop("test")
        train_val_test_dataset["test"] = train_test_dataset["test"]

        return train_val_test_dataset
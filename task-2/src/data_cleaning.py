import pandas as pd
import os

from rich import print

class DataCleaning:
    def __init__(self):
        positive_file_path = '../data/raw/positive_data.csv'
        negative_file_path = '../data/raw/negative_data.csv'
        neutral_file_path = '../data/raw/neural_data.csv'
        files = [positive_file_path, negative_file_path, neutral_file_path]
        self.combined_reviews_df = pd.concat([pd.read_csv(file) for file in files])

    def clean_data(self):
        # Remove NaN columns
        self.combined_reviews_df = self.combined_reviews_df.dropna()
        
        # Reset the index column
        self.combined_reviews_df.reset_index(drop=True)
        num_rows = self.combined_reviews_df.shape[0]
        self.combined_reviews_df['Index'] = range(num_rows)
        self.combined_reviews_df.set_index('Index', inplace=True)
        
        # Fix column typings
        self.combined_reviews_df = self.combined_reviews_df.astype({'Rate': 'float64'})

        # Rename labels so they start from 0
        self.combined_reviews_df["Label"] = self.combined_reviews_df["Label"].map({-1: 0, 0: 1, 1: 2})
        self.combined_reviews_df["Label"] = self.combined_reviews_df["Label"].astype(int)

        # Export cleaned data
        os.makedirs('../data/cleaned', exist_ok=True)
        self.combined_reviews_df.to_csv('../data/cleaned/cleaned_data.csv')
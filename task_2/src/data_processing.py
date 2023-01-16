import pandas as pd

from datasets import Dataset

class DataProcessing:
    def __init__(self, current_directory):
        self.current_directory = current_directory
        data_file_path = f"{self.current_directory}/data/cleaned/cleaned_data.csv"
        self.reviews_df = pd.read_csv(data_file_path)

    def process_data(self):
        """
        The process_data function :
            - Applies lowercasing to the review column.
            - Removes stopwords from the review column.
            - Converts the dataframe into an Arrow table, which is then split into train, val and test sets.
        
        :return: The split dataset
        """
        # Apply lowercasing
        self.reviews_df["Review"] = self.reviews_df["Review"].str.lower()

        # Remove stopwords
        self.reviews_df["Review"] = self.reviews_df["Review"].apply(
            self.remove_vn_stopwords
        )

        # Convert to Arrow format
        reviews_dataset = Dataset.from_pandas(self.reviews_df)

        # Split into train, val, test
        return self.split_train_val_test(reviews_dataset)

    def process_single(self, text: str):
        """
        The process_single function takes a single string as input and returns a single string as output.
        The input is lowercased, stopwords are removed, and the result is returned. It also calls the
        remove_vn_stopwords for the removal of stopwords
        
        :param text: Pass the text to be processed
        :return: The text after removing all the stopwords and lowercased
        """
        # Apply lowercasing
        text = text.lower()
        text = self.remove_vn_stopwords(text)
        return text

    def remove_vn_stopwords(self, text: str):
        """
        The remove_vn_stopwords function references an existing list of Vietnamese stopwords. It then removes
        the stopwords from the text.
        
        :param text:
        :return: The text after stopwords are removed.
        """
        vn_stopwords_file = open(f"{self.current_directory}/data/vietnamese-stopwords.txt", encoding="utf8")
        vn_stopwords = vn_stopwords_file.read()
        vn_stopwords = vn_stopwords.split("\n")

        for stopword in vn_stopwords:
            text = text.replace(stopword, "")
        return text

    def split_train_val_test(self, dataset):
        """
        The split_train_val_test function splits the dataset into three subsets: train, validation and test.
        The split is done by using the train_test_split function from the sklearn library. 
        The split ratio is train: 0.80, test: 0.10, val: 0.10
        
        :param dataset: Pass the dataset to be split
        :return: A dataset with three splits: train, validation and test
        """
        # 1st split; train and (test + val)
        train_test_dataset = dataset.train_test_split(train_size=0.8, shuffle=True, seed=42)

        # 2nd split; split (test + val) equally, add back train dataset
        train_val_test_dataset = train_test_dataset["test"].train_test_split(
            train_size=0.5, shuffle=True, seed=42
        )
        train_val_test_dataset["validation"] = train_val_test_dataset.pop("train")
        train_val_test_dataset["train"] = train_test_dataset["train"]

        return train_val_test_dataset

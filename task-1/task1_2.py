from datasets import load_dataset
from typing import List
from task1_1 import AdjectiveProcessor

import html
import re
import random
import pandas as pd


def get_50_random_sentences() -> List[str]:
    """
    Returns 50 random sentences from the IMDB csv file

    :return: A list of 50 random sentences
    """
    data_files = {"data": "imdb_dataset.csv"}
    imdb_dataset = load_dataset("csv", data_files=data_files)
    imdb_dataset = text_processing(imdb_dataset)

    sentences = []

    for review in imdb_dataset["data"]["review"]:
        # Split into sentences by either fullstop, question mark or exclamation mark, but ignore ellipses
        sentences.extend(re.split(r"(?<!\.)(?<!\!)(?<!\?)[.!?]", review))

    # Ensure that sentences have at least 20 characters, and also remove any blank spaces on the left
    sentences = [sentence.lstrip() for sentence in sentences if len(sentence) >= 20]

    # Get 50 random indexes
    random_numbers = random.sample(range(0, len(sentences) - 1), 50)
    return [sentences[number] for number in random_numbers]


def text_processing(dataset):
    """
    Takes in an Arrow dataset (from datasets module) and does the following processing:
    - Convert any html unicode
    - Remove line breaks
    - Make everything lowercase

    :return: Processed dataset
    """
    # Convert html unicode from reviews
    dataset = dataset.map(lambda x: {"review": html.unescape(x["review"])})
    # Remove line breaks
    dataset = dataset.map(lambda x: {"review": x["review"].replace("<br />", " ")})
    # Set all to lowercase
    dataset = dataset.map(lambda x: {"review": x["review"].lower()})

    return dataset


if __name__ == "__main__":
    sentences = get_50_random_sentences()
    index = 0
    triplet_list = []

    for sentence in sentences:
        adjective_processor = AdjectiveProcessor(sentence)
        triplets = adjective_processor.find_triplets()
        # Extract all triplets from sentences
        if len(triplets) == 0:
            triplet_list.append(
                {
                    "ID": index,
                    "Sentence": sentence,
                    "Adjective": "None",
                    "Noun": "None",
                    "Dependency Relation": "None",
                }
            )
            index += 1
        else:
            for triplet in triplets:
                triplet_list.append(
                    {
                        "ID": index,
                        "Sentence": sentence,
                        "Adjective": triplet["ADJ"],
                        "Noun": triplet["NOUN"],
                        "Dependency Relation": triplet["dependency"],
                    }
                )
                index += 1

    # Populate in excel
    triplet_df = pd.DataFrame(triplet_list)
    triplet_df.to_excel('triplets.xlsx', index=False)

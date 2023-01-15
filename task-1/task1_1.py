from rich import print
from typing import List

import stanza
import argparse


class AdjectiveProcessor:
    def __init__(self, sentence: str):
        """
        Initializes AdjectiveProcessor

        :param sentence: Sentence to be processed
        """
        # stanza.download("en")
        self.nlp = stanza.Pipeline(
            lang="en", processors="tokenize,mwt,pos,lemma,depparse", verbose=False
        )
        self.sentence = sentence

    def find_adjectives(self) -> List[str]:
        """
        The find_adjectives function finds all adjectives in a sentence
        and returns a list of all the adjectives found

        :return: A list of adjectives
        """
        adjectives = []
        doc = self.nlp(self.sentence)
        adjectives = [
            word.text
            for sentence in doc.sentences
            for word in sentence.words
            if word.upos == "ADJ"
        ]
        return adjectives

    def find_triplets(self) -> List[dict]:
        """
        The find_triplets function finds all the triplets in a sentence.
        It returns a list of dictionaries, where each dictionary contains
        an adjective, noun and dependency relation.

        :return: A list of dictionaries
        """
        adjectives = self.find_adjectives()

        triplets = []
        nouns = []

        doc = self.nlp(self.sentence)

        for sentence in doc.sentences:
            for word in sentence.words:
                head_word_text = sentence.words[word.head - 1].text
                # If the word is a noun and the head_word is an adjective,
                # add it to the triplet and add it to the nouns list
                if word.upos == "NOUN" and head_word_text in adjectives:
                    # Append in the form: (adjective, noun, deprel)
                    triplets.append(
                        {
                            "ADJ": head_word_text,
                            "NOUN": word.text,
                            "dependency": word.deprel,
                        }
                    )
                    nouns.append(word.text)
                # If not, add the noun to the nouns list for further processing
                elif word.upos == "NOUN":
                    nouns.append(word.text)

        # Once the nouns list is populated, check if any adjectives
        # refer to the nouns as a head_word. Then add it to the triplets
        for sentence in doc.sentences:
            for word in sentence.words:
                head_word_text = sentence.words[word.head - 1].text
                if word.upos == "ADJ" and head_word_text in nouns:
                    # Append in the form: (adjective, noun, deprel)
                    triplets.append(
                        {
                            "ADJ": word.text,
                            "NOUN": head_word_text,
                            "dependency": word.deprel,
                        }
                    )

        return triplets


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract adjectives and deprel triplets"
    )
    parser.add_argument("--string", type=str, help="string input")
    args = parser.parse_args()

    adjective_processor = AdjectiveProcessor(args.string)
    print("Stanza module loaded!")
    print(f"The sentence is: [magenta]{args.string}[/magenta]")
    print(f"Adjectives are {adjective_processor.find_adjectives()}")
    print(f"Triplets are {adjective_processor.find_triplets()}")

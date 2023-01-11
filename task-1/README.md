## Set-up
- Run `pip install -r requirements.txt` to update/install the libraries needed to run the code
- If you are running the code for the first time, you may need to uncomment line 15 `stanza.download("en")`
- Otherwise, keep it commented out to prevent unnecessary downloading

## Running the code.
- Run `pip install -r requirements.txt` to update/install the libraries needed to run the code
- Think of a sentence and write it in the script as follow:
    - `python task1_1.py --string <YOUR_SENTENCE>
- For example, your sentence may be "My heavy dog vocalizes well into my new microphone which is pink"
- So you will type `python task1_1.py --string "My heavy dog vocalizes well into my new microphone which is pink"`
- The output will be as follows:
    ```
    The sentence is: My heavy dog vocalizes well into my new microphone which is pink
    Adjectives are ['heavy', 'new', 'pink']
    Triplets are [{'ADJ': 'heavy', 'NOUN': 'dog', 'dependency': 'amod'}, {'ADJ': 'new', 'NOUN': 'microphone', 'dependency': 'amod'}, {'ADJ': 'pink', 'NOUN': 'microphone', 'dependency': 'acl:relcl'}]
    ```

### In a paragraph or two, share your thoughts on how the performance of a sentiment analysis classifier can be enhanced by leveraging the triplet above
- From the triplet above, some adjectives would modify the noun to give it a positive, negative or neutral meaning. For example, if we have a sentence like "The movie was bad", then a corresponding triplet may be `{bad, movie, nubj}`. It is likely that similar triplets including the word `bad` would have a negative sentiment as well as `bad` inherently has a negative meaning. The reverse is true for traditionally positive adjectives (for example, `The scene was nice`).
- The model may then give more weight towards adjectives and learn specific adjectives which have inherently positive or negative meanings to give a more accurate sentiment to unseen data. Some adjectives may also inform the model of associations with certain nouns that have positive, negative or neutral connotations. For example `The film was a dreadful slough`. While `dreadful` and `slough` both have negative connotations, the classifier may use the negative association learned from `dreadful` to help associate with `slough`

### Are there any situations in which the code in Deliverable #1 will fail to extract an adjective-noun pair? (i.e. an adjective is modifying a noun but is determined by the parser to not be dependent on the noun) If so, please explain your reasoning in a paragraph or two and give at least one example sentence in which such a failure could occur.
- This can occur if the actual noun is a compound noun, such as `ice cream`. In an example sentence `The new basket ball is bouncy`, in this case, the results are as follows :`[{'ADJ': 'bouncy', 'NOUN': 'ball', 'dependency': 'nsubj'}, {'ADJ': 'new', 'NOUN': 'ball', 'dependency': 'amod'}]`, where basket is ignored in all relations.
- Another case would be compound adjectives, such as `full-time` or `easy-going`. An example `The full-time worker left today` results in the following: `[{'ADJ': 'full', 'NOUN': 'time', 'dependency': 'amod'}]`, where full-time should be treated as the whole adjective.
- Lastly, especially in informal text and speech, we may have sentences containing adjectives that may be referring to nouns in another sentence. Such as `The movie was very good. Compelling, intriguing and captivating`. If we pass just the second sentence `Compelling, intriguing and captivating`, we get no results as all the adjectives are modifying the noun in a previous sentence.
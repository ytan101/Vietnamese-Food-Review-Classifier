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
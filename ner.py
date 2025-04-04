# ner.py

import spacy
import copy

def ner(tokens, labels):
    """ Takes tokens and predictions and uses spaCy to add named entity recognition then returns a list of predicted labels."""

    # Loads multilingual spaCy model
    ml_model = spacy.load("xx_ent_wiki_sm")

    predictions = copy.deepcopy(labels)
    skip_list = ["other", "unk"]

    for i, token_list in enumerate(tokens):

        # Joins tokens to form one string
        text = ' '.join(tokens[i])

        # Processes text with spaCy NER model
        processed_text = ml_model(text)

        for ent in processed_text.ents:

            # Get character span of the named entity
            ent_start = ent.start_char
            ent_end = ent.end_char

            for j, token in enumerate(token_list):
                # Get character span of the token
                token_start = text.find(token)
                token_end = token_start + len(token)

                # Add 'ne' tag if token is inside a named entity span
                if token_start >= ent_start and token_end <= ent_end:
                    if predictions[i][j] not in skip_list:
                        predictions[i][j] = 'ne'

    return predictions



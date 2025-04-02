# ner.py 
# Takes takes tokens and predictions and uses spaCy for named entity recognition.

import spacy

def ner(tokens, predictions):
    # Creates dictionary where the predictions are mapped to the tokens
    token_label_dict = dict(zip(map(tuple, tokens), map(tuple, predictions)))
    
    # Loads spaCy model
    ml_model = spacy.load("xx_ent_wiki_sm")
    
    # Joins all tokens to form one string
    text = " " 
    for item in tokens:
        tweet = ' '.join(item)
        text = text + tweet + "\n"
    
    # Processes text with spaCy NER model
    processed_text = ml_model(text)
    
    # Puts all found named entities into one list.
    named_entities = []
    for word in processed_text.ents:
        named_entities.append(word.text)
    
    # Replaces the label of the token with "ner" if it was found by the NER. 
    for entity in named_entities:
        for key, value in my_dict.items():
            if entity in key:
                my_dict[entity] = "ner"
    
    


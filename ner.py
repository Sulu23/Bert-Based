# ner.py 
# Takes takes tokens and predictions and uses spaCy for named entity recognition.

import spacy

def ner(tokens, labels):       
    # Joins all tokens to form one string and appends all tokens and all labels into one list. 
    text = " " 
    token_list = []
    label_list = []
    for item in tokens:
        tweet = ' '.join(item)
        text = text + tweet + "\n"
        for token in item:
            token_list.append(token)
    for item in labels:
        for label in item:
            label_list.append(label)
            
    # Creates dictionary where the labels are mapped to the tokens
    token_label_dict = dict(zip(token_list, label_list))
    
    # Loads spaCy model
    ml_model = spacy.load("xx_ent_wiki_sm")
    
    # Processes text with spaCy NER model
    processed_text = ml_model(text)
    
    # Puts all found named entities into one list.
    named_entities = []
    for word in processed_text.ents:
        named_entities.append(word.text)
    
    # Replaces the label of the token with "ner" if it was found by the NER. 
    for entity in named_entities:
            if entity in token_label_dict:
                token_label_dict[entity] = 'ner'
    return token_label_dict
    
    


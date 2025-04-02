# main.py
from model import model_code_switching
from preprocessing import tokens_to_sentences, concaternate_tokens
from ner import ner

# start by opening the data
with open("lid_spaeng/debug150.conll") as data:
    #preprocess data
    p_data = tokens_to_sentences(data)

# run the model on the data
predictions, tokens = model_code_switching(p_data)

# preprocessing for next models
predictions, tokens = concaternate_tokens(predictions, tokens)

# labels named entities
ner(tokens, predictions)

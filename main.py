# main.py preprocessing staat er nog in!!
from model import model_code_switching
from preprocessing import tokens_to_sentences, concaternate_tokens

# start by opening the data
with open("lid_spaeng/train.conll") as data:
    #preprocess data
    p_data = tokens_to_sentences(data)

# run the model on the data
predictions, tokens = model_code_switching(p_data)

# post + preprocessing for next models
concaternate_tokens(tokens, predictions)



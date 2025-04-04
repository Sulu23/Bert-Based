# main.py
from model import model_code_switching
from preprocessing import tokens_to_sentences, concatenate_tokens
from ner import ner
from evaluation import evaluate
from extract import extract_true_labels

# start by opening the data
with open("data/dev.conll") as data:
    #preprocess data
    p_data = tokens_to_sentences(data)

# run the model on the data
predictions, tokens = model_code_switching(p_data)

# preprocessing for next models
predictions, tokens = concatenate_tokens(predictions, tokens)

# labels named entities and returns predicted labels
predicted_labels = ner(tokens, predictions)

# extract true labels for evaluation
true_labels = extract_true_labels("data/tsv_dev.conll")

# evaluates the model
evaluate(true_labels, predicted_labels)

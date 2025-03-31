# model.py implement the model
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def model_code_switching(data):
    # The following 2 lines are copies from https://huggingface.co/msislam/code-mixed-language-detection-XLMRoberta
    tokenizer = AutoTokenizer.from_pretrained("msislam/code-mixed-language-detection-XLMRoberta")
    model = AutoModelForTokenClassification.from_pretrained("msislam/code-mixed-language-detection-XLMRoberta")
    print("ok")
    # make empty lists to store all predictions and tokens
    predictions = []
    tokenized = []
    
    for text in data:
        # Get tokens from the model's tokenizer. This line is copied from https://stackoverflow.com/questions/69921629/transformers-autotokenizer-tokenize-introducing-extra-characters
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        # The following 5 lines are copies from https://huggingface.co/msislam/code-mixed-language-detection-XLMRoberta
        inputs = tokenizer(text, add_special_tokens= False, return_tensors="pt")    
        with torch.no_grad():
            logits = model(**inputs).logits
        labels_predicted = logits.argmax(-1)
        lang_tag_predicted = [model.config.id2label[t.item()] for t in labels_predicted[0]]

        # append predictions and tokens from  this tweet to the lists
        predictions.append(lang_tag_predicted)
        tokenized.append(tokens)
    
    return predictions, tokenized

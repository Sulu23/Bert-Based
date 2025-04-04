# evaluation

from sklearn import metrics


def make_flat(nested_lists):
    # Flattens a list of lists
    flat_list = [item for sublist in nested_lists for item in sublist]

    return flat_list


def evaluate(true_labels, predicted_labels):

    # flatten list
    predicted_labels = make_flat(predicted_labels)

    # label that belongs to a troublesome token
    trouble_maker = predicted_labels.pop(4968)


    report = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels, zero_division=0.0)

    print('***** Evaluation *****')
    print(report)

    return

    
    # Precision, recall and F1 score per class
    precision = metrics.precision_score(true_labels, predicted_labels, average=None) 
    recall = metrics.recall_score(true_labels, predicted_labels, average=None) 
    f1 = metrics.f1_score(true_labels, predicted_labels, average=None)
    
    # Macro average scores
    precision_macro = metrics.precision_score(true_labels, predicted_labels, average='macro')
    recall_macro = metrics.recall_score(true_labels, predicted_labels, average='macro')
    f1_macro = metrics.f1_score(true_labels, predicted_labels, average='macro')
    
    print('***** Evaluation *****')
    print("Precision:", precision, "Recall:", recall, "F1 score:", f1, "Precision macro average:", precision_macro, "Recall macro average:", recall_macro, "F1 score macro average:", f1_macro)



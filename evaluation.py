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

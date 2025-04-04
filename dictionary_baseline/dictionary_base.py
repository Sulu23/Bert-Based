from sklearn import metrics


def tokens_labels(file):
    """ Retrieves tokens and labels from data """

    words = []
    labels = []

    for line in file:
        line = line.split()
        words.append(line[0])
        labels.append(line[1])

    return words, labels

def spanish_dict():
    """ constructs and returns set of spanish words """
    spa_dict = set()

    with open('../data/es.txt', 'r') as file3:
        for word in file3:
            spa_dict.add(word.rstrip().lower())

    return spa_dict


def predict(words, spa_dict):
    """ makes rule-based predictions for labels lang2, lang1 and other """
    preds = []

    for word in words:
        if word.lower() in spa_dict:
            preds.append('lang2')
        elif word.isalpha() == False:
            preds.append('other')
        else:
            preds.append('lang1')

    return preds


def evaluate(labels, preds):
    """ evaluates predictions against original labels and prints performance per label """
    evaluation = metrics.classification_report(
            y_true=labels,
            y_pred=preds,
            zero_division=0.0
            )

    print(evaluation)


def main():
    # opens data file
    filepath = '../data/tsv_dev.conll'
    file = open(filepath, 'r')

    # constructs spanish dictionary
    spa_dict = spanish_dict()

    # extracts tokens and labels from data file
    words, labels = tokens_labels(file)

    # closes data file
    file.close()

    # makes predictions
    preds = predict(words, spa_dict)

    # prints and evaluates predictions
    evaluate(labels, preds)


if __name__ == '__main__':
    main()

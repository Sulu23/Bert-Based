# extract.py


def extract_true_labels(tsv_file):
    """ Extract labels from a tsv file and gather them in a list """
    with open(tsv_file, 'r') as f:
        labels = []
        for line in f:
            labels.append(line.split()[1])

    return labels

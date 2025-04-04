# svm.py
from sklearn_svm_classifier import SVMClassifier
import pandas as pd
from sklearn import metrics
import argparse

def read_preprocess(filename):

    with open(filename, mode='r') as f:
        data = []
        for line in f:
            data.append(line.strip().split('\t'))

    df = pd.DataFrame(data, columns=["word", "label"])

    words = df["word"].to_list()
    labels = df["label"].to_list()

    return words, labels


def evaluate(true_labels, predicted_labels):
    print('***** Evaluation *****')

    report = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels, zero_division=0.0)
    print(report)
    report_dict = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels, zero_division=0.0, output_dict=True)

    f1 = report_dict["macro avg"]["f1-score"]

    return f1


def train_svm(words, labels):

    cls = SVMClassifier(kernel='linear')

    print("*** Extracting Features ***")
    train_feats = cls.get_features(words, ngram=3)

    print("*** Applying TF-IDF ***")

    train_feats = cls.tf_idf(train_feats)

    print("*** Training SVM ***")

    cls.fit(train_feats, labels)

    return cls


def test_svm(words, cls):

    print("*** Extracting Features ***")
    test_feats = cls.get_features(words, test_data=True)

    print("*** Applying TF-IDF ***")

    test_feats = cls.tf_idf(test_feats, test_data=True)


    print("*** Making predictions ***")

    predicted_labels = cls.predict(test_feats)

    return predicted_labels

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test SVM model on different datasets.")
    parser.add_argument("--dataset", choices=["full", "small"], default="small", help="Choose dataset: 'full' for full dataset, 'small' for smaller dataset")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # test on full dataset
    if args.dataset == "full":
        print("*** Using full datasets ***\n")
        train_data = "../data/tsv_train.conll"
        test_data = "../data/tsv_dev.conll"

    # test on smaller dataset
    else:
        print("*** Using first 50k lines of datasets ***\n")
        train_data = "../data/tsv_train50k.conll"
        test_data = "../data/tsv_dev50k.conll"

    # train our model
    train_words, train_labels = read_preprocess(train_data)
    cls = train_svm(train_words, train_labels)

    # test our model
    test_words, test_labels = read_preprocess(test_data)
    predictions = test_svm(test_words, cls)


    f1 = evaluate(test_labels, predictions)
    print("f1 score:", f1)




if __name__ == "__main__":
    main()

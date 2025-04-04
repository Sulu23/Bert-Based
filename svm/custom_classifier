import abc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter
import math
from scipy.sparse import hstack

"""
Implement a classifier with required functions:

get_features: should return a feature vector for each sample
 (1-hot, n-hot encodings or etc.)
fit: to train the classifier
predict: to predict test labels
"""


#class CustomClassifier(abc.ABC):
class CustomClassifier():
    def __init__(self):
        self.vectorizer = None
        self.tfidf_transformer = None

    def get_features(self, text_list, ngram=1, test_data=False):
        """ Return word (or ngram) count features for each text as a 2D numpy array """

        if test_data == True:
            features_array = self.vectorizer.transform(text_list)
            return features_array

        self.vectorizer = CountVectorizer(
                                 #    strip_accents = 'unicode',
#                                     strip_accents = 'ascii',
                                     ngram_range=(1, ngram),
 #                                    lowercase=False,
                                 #    lowercase=True,
                                     analyzer='char'
                                     )

        features_array = self.vectorizer.fit_transform(text_list)

        return features_array


    def tf_idf(self, text_feats, test_data=False):

        if test_data == False:
            self.tfidf_transformer = TfidfTransformer().fit(text_feats)

        features_array = self.tfidf_transformer.transform(text_feats)

        return features_array


    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        """ Abstract method to be separately implemented for each custom classifier. """
        pass

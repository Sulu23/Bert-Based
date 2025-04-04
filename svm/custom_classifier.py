# custom_classifier.py
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CustomClassifier():
    def __init__(self):
        self.vectorizer = None
        self.tfidf_transformer = None

    def get_features(self, text_list, ngram=1, test_data=False):
        """ Return ngram count features for each text as a 2D numpy array """

        if test_data == True:
            features_array = self.vectorizer.transform(text_list)
            return features_array

        self.vectorizer = CountVectorizer(
                                     ngram_range=(1, ngram),
                                     analyzer='char'
                                     )

        features_array = self.vectorizer.fit_transform(text_list)

        return features_array


    def tf_idf(self, text_feats, test_data=False):

        if test_data == False:
            self.tfidf_transformer = TfidfTransformer().fit(text_feats)

        features_array = self.tfidf_transformer.transform(text_feats)

        return features_array


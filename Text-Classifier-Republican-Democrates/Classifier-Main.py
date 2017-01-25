"""Assignment 4: Author Profiling

Please add your code where indicated. You may conduct a superficial test of
your code by executing this file in a python interpreter.

This assignment asks you to predict the class membership of a text (e.g., a
Twitter status update) given a corpus of labeled texts. Unlike previous
assignments in which you had to perform a similarity-based analysis, this
assignment asks you to report the probability of class membership.

Your code must address the case when there are *two* classes of interest and
each text is a member of one and only one class. For example, assigning a text
to one of two authors would fit this case.

To evaluate your algorithms you may wish to use the Twitter messages in the
``data/`` directory. Each file in this directory contains the Twitter messages
of one member of the US House of Representatives. The party affiliation of the
member is indicated in the filename. For example,
``data/AustinScottGA08-Georgia-republican.txt`` contains Twitter messages from
a Georgia Republican.

"""
from collections import Counter

def most_frequent_words(text, n, lowercase=False):
    """Return the `n` most frequent words in `text`.

    This is a warmup exercise.

    You may use any tokenizer.

    Args:
        text (str): A string containing space-separated words.
        lowercase (bool, optional): Convert string to lowercase before processing.

    Returns:
        list: The `n` most frequent words.



    """
    # YOUR CODE HERE

    from collections import Counter

    if lowercase:
       words = [word.strip().lower() for word in text.split()]
    else:
       words = [word.strip() for word in text.split()]

    word_count = Counter(words)
    # most_freq = list(word_count.most_common(n))

    most_freq_list = []
    for i,j in word_count.most_common(n):
        most_freq_list.append(i)

    return most_freq_list

    pass


def most_frequent_bigrams(text, n, lowercase=False):
    """Return the `n` most frequent word bigrams in `text`.

    A word bigram is a sequence of two adjacent words. The following word
    bigrams are present in the string ``it is a truth universally
    acknowledged``: ``{'it is', 'is a', 'a truth', 'truth universally',
    'universally acknowledged'}``.

    You may use any tokenizer.

    Args:
        text (str): A string containing space-separated words.
        lowercase (bool, optional): Convert string to lowercase before processing.

    Returns:
        list: The `n` most frequent word bigrams.

    """
    # YOUR CODE HERE

    from collections import Counter

    if lowercase:
       words = [word.strip().lower() for word in text.split()]
    else:
       words = [word.strip() for word in text.split()]

    bigrams = list(zip(words,words[1:]))
    bi_count = Counter(bigrams)

    most_freq_biagram = []

    for i,j in bi_count.most_common(n):
        most_freq_biagram.append(i)

    return most_freq_biagram

    pass


def accuracy(labels, labels_true):
    """Calculate the percentage of correct labels provided by a binary classifier.

    One simple method of evaluating a binary classifier is to report its
    accuracy at classifying items with known labels. Accuracy in this case is a
    fraction: the number of correct labels divided by the total number of
    items.

    This function need only work for the binary classification case. You may
    rely on there being only two distinct labels in `labels` and `labels_true`.

    Args:
        labels (list): A list of labels. Labels may be integers or strings.
        labels_true (list): A list of expected labels. Labels may be integers or strings.

    Returns:
        float: accuracy

    """
    # YOUR CODE HERE

    total_label = len(labels)
    correct_label = 0

    for i in range(total_label):
        if labels[i] == labels_true[i]:
           correct_label += 1

    return correct_label/total_label
    pass


def predict_label(texts, labels, text_new):
    """Predict label of `text_new` given labeled examples.

    Each text (a string) in `texts` is associated with one of two labels (also
    a string). Using these examples, predict the label of an unlabeled text,
    `text_new`.

    For example, suppose you are given a collection of English-language
    Wikipedia edits and the provided label indicates whether or not the edit
    was made by a native or non-native speaker of English. (The labels would be
    the strings ``native`` or `non-native``.) Given an unlabeled edit,
    `text_new`, report ``native`` if it is an edit by a native speaker or
    ``non-native`` if it is an edit by a non-native speaker.

    Args:
        texts (list of str): A list of texts.
        labels (list): A list of labels corresponding to the elements of `texts`.
        text_new (str): A string containing an unlabeled text.

    Returns:
        str: The predicted label of `text_new`.

    """
    # YOUR CODE HERE

    # texts = ['RT @GOPLeader', 'RT @GOPLeader', 'Colorless green ideas sleep furiously.']
    # labels = ['rep', 'rep', 'dem']

    train_twitter = texts
    test_twitter = text_new

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB

    count_vect = CountVectorizer()
    twitter_train_counts = count_vect.fit_transform(train_twitter)

    tf_transformer = TfidfTransformer(use_idf=False).fit(twitter_train_counts)
    twitter_train_tf = tf_transformer.transform(twitter_train_counts)

    tfidf_transformer = TfidfTransformer()
    twitter_train_tfidf = tfidf_transformer.fit_transform(twitter_train_counts)

    twitter_clf = MultinomialNB().fit(twitter_train_tfidf,labels )

    # transforming the test data

    twitter_test_data = count_vect.transform(test_twitter)
    twitter_tfidf = tfidf_transformer.transform(twitter_test_data)

    #prediction
    twitter_predicted = twitter_clf.predict(twitter_tfidf)

    for text, class_label in zip(test_twitter, twitter_predicted):
        print('%r => %s' % (text, class_label))


    return list(twitter_predicted)




def predict_label_naive(texts, labels, text_new):
    """Example of (naive) prediction strategy: predict most frequent label.

    Your function `predict_label` should aspire to perform better than this on
    average.

    """
    counter = {label: 0 for label in sorted(set(labels))}
    for label in labels:
        counter[label] += 1
    most_frequent_count = max(counter.values())
    most_frequent_labels = [key for key, value in counter.items() if value == most_frequent_count]
    # if labels equally frequent, return whichever is first
    return most_frequent_labels.pop()


# RECOMMENDED EXERCISES
#
# The following exercises are recommended.
#


def log_loss(labels_probability, labels_true):
    """Calculate the log loss of a binary classifier.

    The log loss is also known as the cross-entropy loss. It is defined
    to be:

        $$J(p, y) = - \sum_{i=1}^n y_i \log(p_i) + (1 - y_i) \log(1-p_i)$$

    where $y_i$ is the true class (0 or 1) and $p_i$ is the predicted
    probability of the class being 1.

    Args:
        labels_probability (list of float): A list of probabilities of the label `1` being assigned.
        labels_true (list of int): A list of integers. Each element must be either 0 or 1.

    Returns:
        float: The log loss (aka cross-entropy loss)

   """

    from sklearn.metrics import log_loss as ll

    return ll(y_true = labels_true,y_pred = labels_probability,labels = [1,0],normalize = False)


def predict_label_probability(texts, labels, text_new):
    """Predict probability of class membership for `text_new`.

    Each text (a string) in `texts` is associated with one of two labels (also
    a string). Using these examples, return the probability of the unlabeled text
    receiving the "positive" label (here the integer 1).

    For example, suppose you are given a collection of English-language
    Wikipedia edits and the provided label indicates whether or not the edit
    was made by a native or non-native speaker of English. (The labels would be
    the integers 0 (for ``native``)  or 1 (for `non-native``).) For example,
    given an unlabeled edit, `text_new`, report ``0.9`` if the probability that
    the text is from a ``non-native`` speaker of English is 90%.

    NOTE: A flexible version of this function would accept string or integer
    labels and associate the positive case with the highest value (if integer)
    or the lexicographically last value (if string). (This does assume there are at least
    two distinct strings or integers.)

    Args:
        texts (list of str): A list of texts.
        labels (list of int): A list of integer labels corresponding to the elements of `texts`.
        text_new (str): A string containing an unlabeled text.

    Returns:
        float: The probability of `text_new` receiving the positive label.

    """

    train_twitter = texts
    test_twitter = text_new

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    count_vect = CountVectorizer()
    twitter_train_counts = count_vect.fit_transform(train_twitter)

    tf_transformer = TfidfTransformer(use_idf=False).fit(twitter_train_counts)
    twitter_train_tf = tf_transformer.transform(twitter_train_counts)


    tfidf_transformer = TfidfTransformer()
    twitter_train_tfidf = tfidf_transformer.fit_transform(twitter_train_counts)

    twitter_clf = LogisticRegression().fit(twitter_train_tfidf,labels)

    twitter_test_data = count_vect.transform(test_twitter)
    twitter_tfidf = tfidf_transformer.transform(twitter_test_data)


    twitter_predicted = twitter_clf.predict(twitter_tfidf)

    for text, class_label in zip(test_twitter, twitter_predicted):
        print('%r => %s' % (text, class_label))


    class_prob = list(twitter_clf.predict_proba(twitter_tfidf)[:,1])

    return class_prob
    pass



# DO NOT EDIT CODE BELOW THIS LINE

import unittest

import numpy as np


class TestAssignment04(unittest.TestCase):

    def test_most_frequent_words(self):
        text = 'the cat in the hat'
        mfw = most_frequent_words(text, 1)
        self.assertEqual(mfw, ['the'])
        self.assertEqual(len(most_frequent_words(text, 2)), 2)
        self.assertEqual(len(most_frequent_words(text, 3)), 3)

    def test_most_frequent_bigrams(self):
        text = 'it is a truth universally acknowledged'
        self.assertEqual(len(most_frequent_bigrams(text, 1)), 1)
        self.assertEqual(len(most_frequent_bigrams(text, 2)), 2)
        self.assertEqual(len(most_frequent_bigrams(text, 3)), 3)

    def test_accuracy(self):
        acc = accuracy(['dem', 'rep', 'rep', 'rep'],
                       ['rep', 'rep', 'rep', 'rep'])
        self.assertEqual(acc, 0.75)

    def test_log_loss(self):
        loss = log_loss([0.5, 0.5, 0.5, 0.5],
                        [1, 1, 1, 1])
        self.assertEqual(loss, -1 * np.log(0.5) * 4)
        loss = log_loss([0.1, 0.5, 0.5, 0.5],
                        [0, 1, 1, 1])
        self.assertEqual(loss, -1 * (np.log(0.5) * 3 + np.log(0.9)))

    def test_predict_label_probability_naive(self):
        texts = ['RT @GOPLeader', 'RT @GOPLeader', 'Colorless green ideas sleep furiously.']
        labels = ['rep', 'rep', 'dem']
        text_new = 'RT'
        self.assertEqual(predict_label_naive(texts, labels, text_new), 'rep')


if __name__ == '__main__':
    unittest.main()

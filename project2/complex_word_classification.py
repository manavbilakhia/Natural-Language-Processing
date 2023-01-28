"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Manav Bilakhia

I affirm that I have carried out my academic endeavors with full
academic honesty. [Manav Bilakhia]

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    _,labels = load_file(data_file)
    y_pred = [1 for label in labels]
    evaluate(y_pred, labels)


### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    training_data = load_file(training_file)
    train_words = training_data[0]
    train_labels = training_data[1]

    development_data = load_file(development_file)
    dev_words = development_data[0]
    dev_labels = development_data[1]

    best_threshold = 0
    best_fscore = 0
    y_pred_train = []
    for threshold in range(1, 20):
        y_pred_train = []
        for word in train_words:
            if len(word) >= threshold:
                y_pred_train.append(1)
            else:
                y_pred_train.append(0)
        fscore = get_fscore(y_pred_train, train_labels)
        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold
    y_pred_dev = []
    for word in dev_words:
        if len(word) >= best_threshold:
            y_pred_dev.append(1)
        else:
            y_pred_dev.append(0)
    print("Best threshold:", best_threshold)
    evaluate(y_pred_dev, dev_labels)

### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    min_freq = min(counts.values())
    max_freq = max(counts.values())
    print("minimum frequency:", min_freq)
    print("maximum frequency:", max_freq)

    freq_thresholds = range (min_freq, max_freq, 100000)
    best_threshold = 0
    best_fscore = 0
    training_data = load_file(training_file)


### 3.1: Naive Bayes

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    ## YOUR CODE HERE
    pass


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    ## YOUR CODE HERE
    pass


### 3.3: Build your own classifier

def my_classifier(training_file, development_file, counts):
    ## YOUR CODE HERE
    pass


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)

if __name__ == "__main__":
    #all_complex("train/complex_words_training.txt")
    word_length_threshold("train/complex_words_training.txt", "train/complex_words_development.txt")
    #training_file = "train/complex_words_training.txt"
    #development_file = "train/complex_words_development.txt"
    #test_file = "train/complex_words_test_unlabeled.txt"

    #print("Loading ngram counts ...")
    # ngram_counts_file = "ngram_counts.txt.gz"
    # counts = load_ngram_counts(ngram_counts_file)

    # baselines(training_file, development_file, counts)
    # classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE




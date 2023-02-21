"""Named Entity Recognition as a classification task.

Author: Kristina Striegnitz and Manav Bilakhia

I affirm that I have carried out my academic endeavors with full
academic honesty. [Manav Bilakhia]


This file contains the Named Entity Recognition (NER) system. 
It implements a classification based approach to NER.
"""
import nltk
from nltk.corpus import conll2002
nltk.data.path.append('C:\ManavData\college\Courses\CSC483-NLP\CSC483-NLP_projects\project3')
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def getfeats(word, o):
    """
    Take a word and its offset with respect to the word we are trying
    to classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        (o + 'is_upper', word.isupper()),
        (o + 'is_lower', word.islower()),
        (o + 'is_title', word.istitle()),
        (o + 'is_digit', word.isdigit()), 
         # check if word contains an apostrophe
        #(o + 'contains_apostrophe', "'" in word) # no change
    ]
    if len(word) > 1:
        features.extend([
            #(o + 'ends_with_s', word.endswith('s')),#60.74 -> 61.14
            #(o + 'ends_with_ly', word.endswith('ly')),# no change
            (o + 'ends_with_ing', word.endswith('ing'))
            #(o + 'ends_with_able', word.endswith('able')) #60.72->60.74
        ])
    #if '-' in word: # increased 60.64->60.72
    #    features.append((o + 'word.hyphenated', True))
    #if word.endswith('mente'): # no change
    #    features.append((o + 'word.adverb', True))
    return features
    

def word2features(sent, i):
    """Generate all features for the word at position i in the
    sentence. The features are based on the word itself as well as
    neighboring words.
    """
    features = []
    # the window around the token (o stands for offset)
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features

if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.
    
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, train_labels)

    print("\nTesting ...")
    test_feats = []
    test_labels = []

    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_classifier.txt. That is the results_classifier.txt you
    # should hand in.
    for sent in test_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    # If you are normaling and/or scaling your training data, make
    # sure to transform your test data in the same way.
    y_pred = model.predict(X_test)

    print("Writing to results_classifier.txt")
    # format is: word gold pred
    j = 0
    with open("results_classifier.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_classifier.txt")
"""Named Entity Recognition as a classification task.

Author: Kristina Striegnitz and Manav Bilakhia

I affirm that I have carried out my academic endeavors with full
academic honesty. [Manav Bilakhia]


THis file contains the Named Entity Recognition (NER) system.
It implements the Viterbi algorithm for NER with a MEMM.
"""
import nltk
from nltk.corpus import conll2002
nltk.data.path.append('C:\ManavData\college\Courses\CSC483-NLP\CSC483-NLP_projects\project3')
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import math
import numpy as np
# TODO (optional): Complete the class MEMM
from memm import MEMM

#################################
#
# Word classifier
#
#################################

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
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features


#################################
#
# Viterbi decoding
#
#################################

def viterbi(obs, memm, pretty_print=False):
    # TODO: complete this function. Implement an adapted version of
    # the viterbi algorithm that decodes based on the memm's
    # classifier's output.

    # Initialize the trellis
    trellis = [{}] # trellis[i][state] = {"score": score, "backpointer": backpointer}
    for state in memm.states:
        features = word2features(obs, 0)
        features["prev_tag"] = "<S>"
        probs = memm.classifier.predict_log_proba(memm.vectorizer.transform(features))[0] # probs = [log(p1), log(p2), ...]
        score = probs[memm.states.index(state)]
        trellis[0][state] = {"score": score, "backpointer": None} # trellis[0][state] = {"score": score, "backpointer": None}

    # Fill in the trellis
    for i in range(1, len(obs)):
        trellis.append({})  
        for state in memm.states:
            max_score = float("-inf") #we choose the max score as -inf because we are using log probabilities
            max_prev_tag = None
            features = word2features(obs, i)
            for prev_state in memm.states:
                prev_score = trellis[i-1][prev_state]["score"] #get the score of the previous state
                features["prev_tag"] = prev_state
                probs = memm.classifier.predict_log_proba(memm.vectorizer.transform(features))[0]
                score = prev_score + probs[memm.states.index(state)] #calculate the score of the current state
                if score > max_score:
                    max_score = score
                    max_prev_tag = prev_state
            trellis[i][state] = {"score": max_score, "backpointer": max_prev_tag}   

    # Find the highest-scoring path
    best_path = []
    current_tag = max(trellis[-1], key=lambda state: trellis[-1][state]["score"])
    best_path.append(current_tag) #append the last tag to the best path
    for i in range(len(obs)-1, 0, -1):
        current_tag = trellis[i][current_tag]["backpointer"] #get the backpointer of the current tag
        best_path.append(current_tag)
    best_path.reverse() #reverse the best path

    # Print the trellis if requested
    if pretty_print:
        for i in range(len(obs)):
            print(f"Observation {i}: {obs[i]}")
            for state in memm.states:
                print(f"\t{state}: score={trellis[i][state]['score']:.2f}, backpointer={trellis[i][state]['backpointer']}") 
    return best_path


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
            # TODO: training needs to take into account the label of
            # the previous word. And <S> if i is the first words in a
            # sentence.
            if i == 0:
                feats["prev_tag"] = "<S>"  # the previous tag is <S> if i is the first word in a sentence
            else:
                feats["prev_tag"] = sent[i-1][-1] # the previous tag is the tag of the previous word
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
    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_memm.txt. That is the results_memm.txt you should hand
    # in.
    y_pred = []
    for sent in test_sents:
        # TODO: extract the feature representations for the words from
        # the sentence; use the viterbi algorithm to predict labels
        # for this sequence of words; add the result to y_pred
        feats = []
        for i in range(len(sent)):
            feats.append(dict(word2features(sent,i))) # extract the feature representations for the words from the sentence
            if i == 0:
                feats[i]["prev_tag"] = "<S>" # the previous tag is <S> if i is the first word in a sentence
            else:
                feats[i]["prev_tag"] = sent[i-1][-1] # the previous tag is the tag of the previous word
        X_test = vectorizer.transform(feats) # transform the features into vectors of numbers
        y_pred.extend(model.predict(X_test)) # predict the labels for the sequence of words and add the result to y_pred

    print("Writing to results_memm.txt")
    # format is: word gold pred
    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in test_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_memm.txt")

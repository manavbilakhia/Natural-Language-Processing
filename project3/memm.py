"""
Author: Kristina Striegnitz and Manav Bilakhia

I affirm that I have carried out my academic endeavors with full
academic honesty. [Manav Bilakhia]

This class implements a Maximum Entropy Markov Model (MEMM) for
sequence labeling. The MEMM is trained using the Viterbi algorithm.
"""

class MEMM:
    def __init__(self, states, vocabulary, vectorizer, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.states = states
        self.vocabulary = dict((vocabulary[i], i) for i in range(len(vocabulary)))
        self.vectorizer = vectorizer
        self.classifier = classifier


    # TODO: Add additional methods that are needed. In particular, you
    # will need a method that can take a dictionary of features
    # representing a word and the tag chosen for the previous word and
    # return the probabilities of each of the MEMM's states.
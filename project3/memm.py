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

    def get_feature_probabilities(self, features, prev_tag):
        """Given a dictionary of features representing a word and the tag chosen
        for the previous word, return the probabilities of each of the MEMM's states.

        Args:
        - features: a dictionary of features representing a word
        - prev_tag: the tag chosen for the previous word

        Returns:
        - probabilities: a dictionary mapping each state to its probability
        """
        # Add previous tag information to the features
        features['prev_tag'] = prev_tag

        # Convert features to a feature vector
        feature_vector = self.vectorizer.transform(features)

        # Get the predicted log probabilities for each state
        log_probabilities = self.classifier.predict_log_proba(feature_vector)

        # Convert log probabilities to probabilities
        probabilities = {}
        for i in range(len(self.states)):
            probabilities[self.states[i]] = math.exp(log_probabilities[0][i])

        return probabilities
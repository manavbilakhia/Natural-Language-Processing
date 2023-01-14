import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    text = start_pad(c) + text
    return [(text[i:i+c], text[i+c]) for i in range(len(text)-c)]


def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        ''' Initializes the n-gram model with the context length c and the
            smoothing parameter k '''
        self.c = c
        self.k = k
        self.ngrams = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.ngrams.keys())

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, char in ngrams(self.c, text):
            if context not in self.ngrams:
                self.ngrams[context] = {}
            if char not in self.ngrams[context]:
                self.ngrams[context][char] = 0
            self.ngrams[context][char] += 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.ngrams:
            return 0
        context_count = sum(self.ngrams[context].values()) + self.k * len(self.ngrams[context])
        char_count = self.ngrams[context].get(char, 0) + self.k
        return char_count / context_count

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        if context not in self.ngrams:
            return None
        context_count = sum(self.ngrams[context].values()) + self.k * len(self.ngrams[context])
        char_probs = {}
        for char in self.ngrams[context]:
            char_probs[char] = (self.ngrams[context][char] + self.k) / context_count
        r = random.random()
        p_total = 0.0
        for char, p in char_probs.items():
            p_total += p
            if r < p_total:
                return char

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = " " * (self.c - 1)
        text = ""
        for i in range(length):
            char = self.random_char(context)
            if char is None:
                return None
            text += char
            context = context[1:] + char
        return text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        n = len(text)
        context = " " * (self.c - 1)
        prob = 1.0
        for i in range(n):
            char = text[i]
            p = self.prob(context, char)
            if p == 0:
                return float('inf')
            prob *= (1/p)
            context = context[1:] + char
        return pow(prob, -1/n)

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        ''' Initializes the n-gram model with the context length c and the
            smoothing parameter k '''
        pass
    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        pass
    def update(self, text):
        ''' Updates the model n-grams based on text '''
        pass

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        pass

################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.

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
    ngrams_list = []
    for i in range(c, len(text)):
        context = text[i-c:i] if c > 0 else ""
        char = text[i]
        ngrams_list.append((context, char))
    return ngrams_list


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
        self.vocab = set()

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, char in ngrams(self.c, text):
            self.vocab.add(char)
            if context not in self.ngrams:
                self.ngrams[context] = {}
            if char not in self.ngrams[context]:
                self.ngrams[context][char] = 0
            self.ngrams[context][char] += 1

    #def prob(self, context, char):
    #    ''' Returns the probability of char appearing after context '''
    #    if context not in self.ngrams:
    #        return 1/len(self.vocab)
    #    context_count = sum(self.ngrams[context].values())
    #    if char not in self.ngrams[context]:
    #        return 0.0
    #    char_count = self.ngrams[context][char]
    #    return char_count/context_count
    
    def prob(self, context, char):
        ''' Returns the probability of char appearing after context with add-k smoothing'''
        if context not in self.ngrams:
            return 1/len(self.vocab)
        context_count = sum(self.ngrams[context].values()) + self.k * len(self.vocab)
        char_count = self.ngrams[context].get(char, 0) + self.k
        return char_count/context_count

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        if context not in self.ngrams:
            return random.choice(list(self.vocab))
        prob_sum = 0
        r = random.random()
        vocab = sorted(self.vocab)
        for char in vocab:
            prob_sum += self.prob(context, char)
            if r < prob_sum:
                return char
    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        if self.c == 0:
            context = ""
        else:
            context = start_pad(self.c)
        result = ""
        for i in range (length):
            char = self.random_char(context)
            result += char
            context = context[1:] + char
        return result
        
    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        if self.c == 0:
            context = ""
        log_prob = 0
        for context, char in ngrams(self.c, text):
            p = self.prob(context, char)
            if p == 0:
                return float("inf")
            log_prob += math.log(self.prob(context,char))
        return math.exp(-log_prob/len(text))

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

if __name__ == '__main__':
    print(ngrams(1,"abc"))
    print(ngrams(2,"abc"))
    m = NgramModel(1, 0)
    m.update("abab")
    vocab1 = m.get_vocab()
    print(vocab1)
    m.update("abcd")
    vocab2 = m.get_vocab()
    print(vocab2)
    print(m.prob("a", "b"))
    print(m.prob("~", "c"))
    print(m.prob("b", "c"))
    m = NgramModel(0, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print([m.random_char('') for i in range(25)])
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print(m.random_text(25))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 2)
    print(m.random_text(250))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 3)
    print(m.random_text(250))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 4)
    print(m.random_text(250))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 7)
    print(m.random_text(250))
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity("abcd"))
    print(m.perplexity("abca"))
    print(m.perplexity("abcda"))
    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob("a","a"))
    print(m.prob("a","b"))
    print(m.prob("c","d"))
    print(m.prob("d","a"))
    print("------------------")
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print(m.prob("a","a"))
    print(m.prob("a","b"))
    print("------------------")
    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob("~a","b"))
    print(m.prob("ba","b"))
    print(m.prob("~c","d"))
    print(m.prob("bc","d"))
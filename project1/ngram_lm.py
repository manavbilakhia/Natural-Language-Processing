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
        super().__init__(c, k)
        self.lambdas = [1/(c+1) for _ in range(c+1)]
        self.models = [NgramModel(i, k) for i in range(c)]

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        super().update(text)
        for model in self.models:
            model.update(text)
    
    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        model_sum = 0
        for model in self.models:
            model_prob = 0
            model_char = ngrams(model.c, context + char)[-1][1]
            model_context = ngrams(model.c, context + char)[-1][0]
            model_prob = model.prob(model_context, model_char)
            model_sum += self.lambdas[model.c+1] * model_prob
        model_sum = model_sum+ (self.lambdas[0] * super().prob(context, char))
        return model_sum    

    def set_lambdas(self, lambdas):
        ''' Sets the interpolation weights to lambdas '''
        self.lambdas = lambdas
################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.
class language_identification():
    #James Heffernan helped me with the algorithm for this class
    ''' A language identification model '''
    def __init__(self, lambdas):
        self.models = {}
        for code in COUNTRY_CODES:
            path = "cities_train/train/"+code+".txt"
            model = create_ngram_model_lines(NgramModelWithInterpolation,path, c = 4,k = 0.5)
            #model.set_lambdas(lambdas)
            self.models[code] = model
    
    def identify_word(self, text):
        ''' Returns the language code of the language that the text is most likely to be in '''
        probability_code = []
        for code,model in self.models.items():
            probability = 0
            for context, char in ngrams(model.c, text):
                probability += math.log(model.prob(context, char))
            probability_code.append((probability, code))
        return max(probability_code)[1]
    
    def classifier_accuracy(self,code):
        ''' Returns the accuracy of the model on the test data for the given language code '''
        
        countries = []
        with open('cities_val/val/'+code+'.txt') as file:
            for line in file:
                line = line.strip()
                countries.append(self.identify_word(line))
        return countries.count(code)/len(countries)
    
    def evaluate(self):
        ''' Returns the accuracy of the model on the test data for all languages'''
        for code in COUNTRY_CODES:
            accuracy = self.classifier_accuracy(code)
            percentage = accuracy*100
            print(code,": ",percentage,"%")
def test1():
    print(ngrams(1,"abc"))
    print(ngrams(2,"abc"))

def test2():
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

def test3():
    m = NgramModel(0, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print([m.random_char('') for i in range(25)])

def test4():
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print(m.random_text(25))

def test_shakespear():
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 2)
    print(m.random_text(250))
    print("------------------")
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 3)
    print(m.random_text(250))
    print("------------------")
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 4)
    print(m.random_text(250))
    print("------------------")
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 7)
    print(m.random_text(250))
    print("------------------")

def test_perplexity():
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity("abcd"))
    print(m.perplexity("abca"))
    print(m.perplexity("abcda"))

def test_shakespeare_perplexity(path):
    models = [create_ngram_model(NgramModel, 'shakespeare_input.txt', c=c+1, k=1) for c in range(10)]
    with open(path, encoding='utf-8', errors='ignore') as file:
        text = file.read()
    for m in models:
        print("c =" , str(m.c), " = " f'{m.perplexity(text):.4f}')

def test_smoothing():
    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob("a","a"))
    print(m.prob("a","b"))
    print(m.prob("c","d"))
    print(m.prob("d","a"))

def test_interpolation():
    m1 = NgramModelWithInterpolation(1, 0) #lambda = [0.5,0.5]
    m1.update('abab')
    print(m1.prob("a","a"))
    print(m1.prob("a","b"))
    print("----------------")
    m2 = NgramModelWithInterpolation(2, 1) #lambda = [1/3,1/3,1/3]
    m2.update('abab')
    m2.update('abcd')
    print(m2.prob("~a","b"))
    print(m2.prob("ba","b"))
    print(m2.prob("~c","d"))
    print(m2.prob("bc","d"))
    print("----------------")
    m3 = NgramModelWithInterpolation(2, 1) #lambda = [2/3,1/6,1/6]
    m3.set_lambdas([2/3,1/6,1/6])
    m3.update('abab')
    m3.update('abcd')
    print(m3.prob("~a","b"))
    print(m3.prob("ba","b"))
    print(m3.prob("~c","d"))
    print(m3.prob("bc","d"))
    print("----------------")
    m4 = NgramModelWithInterpolation(2, 1) #lambda = [0,2/3,1/6]
    m4.set_lambdas([0,2/3,1/6])
    m4.update('abab')
    m4.update('abcd')
    print(m4.prob("~a","b"))
    print(m4.prob("ba","b"))
    print(m4.prob("~c","d"))
    print(m4.prob("bc","d"))

def test_language_identification():
    m = language_identification([0.1, 0.2,0.3,0.4])
    m.evaluate()


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test_shakespear()
    test_perplexity()
    test_shakespeare_perplexity("shakespeare_sonnets.txt")
    test_shakespeare_perplexity("nytimes_article.txt")
    test_smoothing()
    test_interpolation()
    test_language_identification()
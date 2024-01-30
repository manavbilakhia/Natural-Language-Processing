"""This class uses POS to create blanks in a given string

Author: <Saeed AlSuwaidi>

<I affirm that I have carried out my academic endeavors with full academic honesty. [Saeed AlSuwaidi]>

"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from details import Details

class TextWithBlanks:
    """"""
    def __init__(self, text, blanks):
        """
        A class that represents a text with blanks
        :param text: A string of the text where each blank is ___ (e.g I loved the movie ___)
        :param blanks: A list of labels for all the blanks (e.g ["PROPER_NOUN"])
        """
        self.text = text
        self.blanks = blanks

    def __str__(self):
        result = ""
        prev_end = 0
        for (start, end, blank) in self.blanks:
            result += self.text[prev_end:start] + blank
            prev_end = end

        return result


class BlanksCreator:
    def create_blanks(self, text):
        """
        Takes in text and outputs text with blanks where the named entities were found
        :param text: a generated text to create blanks for
        :return: TextWithBlanks
        """
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        
        # use pipeline function
        tag_list = nlp(text)

        names = []
        people = []
        locations = []

        start = None
        end = None
        current_tag = None

        blanks = []

        for tag in tag_list:
            entity = tag['entity']
            if start is not None and entity[:2] == 'B-':
                thing = text[start:end]

                blank = ""
                if current_tag == 'B-MISC':
                    if thing in names:
                        i = names.index(thing)
                    else:
                        i = len(names)
                        names.append(thing)

                    blank = '[NAME-' + str(i) + ']'
                elif current_tag == 'B-PER':
                    if thing in people:
                        i = people.index(thing)
                    else:
                        i = len(people)
                        people.append(thing)

                    blank = '[PER-' + str(i) + ']'
                elif current_tag == 'B-LOC':
                    if thing in locations:
                        i = locations.index(thing)
                    else:
                        i = len(locations)
                        locations.append(thing)

                    blank = '[LOC-' + str(i) + ']'

                blanks.append((start, end, blank))

                current_tag = None
                start = None
                end = None

            if entity == 'B-MISC' or entity == 'B-PER' or entity == 'B-LOC':
                start = tag['start']
                end = tag['end']
                current_tag = tag['entity']
            if start is not None and entity[:2] == 'I-':
                end = tag['end']

        return TextWithBlanks(text, blanks)


# example usage
if __name__ == '__main__':
    creator = BlanksCreator()
    example = "Fast & Furious 6 (2013) is the best of the best Action film in the series franchise! It's Justin Lin's Masterpiece and I love it to death. Sorry but is not Fast Five and Furious 7 the best, but is Fast & Furious 6 the best one in the franchise! This my favorite best film of the franchise that I just love to death! The film has great cast and great action, great dialogue! It is the last time that Paul Walker starts in this film. In Furious 7 was his brother who was portraying Brian O'Conner with a fake CGI. In here Paul Walker is real, Justin Lin started a great franchise which ended in this film."
    new_text = creator.create_blanks(example)
    print(new_text)  # Output: "My name is [PER] [PER] and I live in [LOC] [LOC], My favorite movie is '[MISC] [MISC] [MISC]' by [ORG] lol, but I live in [LOC]"


'''@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
'''
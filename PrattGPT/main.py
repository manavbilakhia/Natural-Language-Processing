from generator import MovieReviewGenerator
from blanks_creator import BlanksCreator
from synopsis_extractor import SynopsisExtractor


def main():
    synopsis = ""

    g = MovieReviewGenerator()
    text = g.generate()

    b = BlanksCreator()
    text_with_blanks = b.create_blanks(text)

    e = SynopsisExtractor()
    e.extract(synopsis, text_with_blanks)

    # TODO: some code to fill in each blank


if __name__ == '__main__':
    #test change to learn git
    main()

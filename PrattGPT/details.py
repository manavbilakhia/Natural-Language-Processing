class Details:
    def __init__(self, text, tag_list):
        names = set()
        people = set()
        locations = set()

        start = None
        end = None
        current_tag = None
        for tag in tag_list:
            entity = tag['entity']
            if start is not None and entity[:2] == 'B-':
                thing = text[start:end]

                if current_tag == 'B-MISC':
                    names.add(thing)
                elif current_tag == 'B-PER':
                    people.add(thing)
                elif current_tag == 'B-LOC':
                    locations.add(thing)

                current_tag = None
                start = None
                end = None

            if entity == 'B-MISC' or entity == 'B-PER' or entity == 'B-LOC':
                start = tag['start']
                end = tag['end']
                current_tag = tag['entity']
            if start is not None and entity[:2] == 'I-':
                end = tag['end']

        self.names = list(names)
        self.people = list(people)
        self.locations = list(locations)

import spacy


class NERPipeline:

    def __init__(self,
                 model_name='en_ner_bionlp13cg_md'):
        nlp = spacy.load(model_name)
        self.nlp = nlp

    def find_entities(self,
                      text):
        nlp = self.nlp
        doc = nlp(text)
        entities = doc.ents
        return entities

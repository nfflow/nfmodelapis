import spacy


class NERPipeline:

    def __init__(self,
                 data,
                 model_name='en_ner_bionlp13cg_md'):
        nlp = spacy.load(model_name)
        self.nlp = nlp
        self.data = data

    def find_entities(self,
                      text):
        nlp = self.nlp
        doc = nlp(text)
        entities = doc.ents
        return entities

    def batch_ner(self, context_col):
        data = self.data
        res_list = []
        for c in data[context_col]:
            ents = self.find_entities(c)
            res_list.append(ents)
        return res_list

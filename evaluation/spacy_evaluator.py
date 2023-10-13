import spacy

class SpacyEvaluator():

    def __init__(self, similarity_threshold=0.8, model='en_core_web_lg') -> None:
        self.nlp = spacy.load(model)
        self.similarity_threshold = similarity_threshold
    
    def similarity(self, doc1: str, doc2: str):
        nlp_doc1 = self.nlp(doc1)
        nlp_doc2 = self.nlp(doc2)
        return nlp_doc1.similarity(nlp_doc2)

    def is_relevant(self, doc1: str, doc2: str):
        simularity = self.similarity(doc1, doc2)
        print(f"Simularity Score: {simularity}")

        if simularity > self.similarity_threshold:
            return True
        
        return False

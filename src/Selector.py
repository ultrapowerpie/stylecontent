from collections import Counter, defaultdict as dd
from GloVe import GloVe
from Generator import Generator
import util as u
import math, random

class Selector:

    def __init__(self, content_file, gen_file):

        with open(content_file, "r") as f:
            self.content = f.read()

    def translate(self, glove):
        '''
        :Args:      GloVe object of word embeddings

        :returns:   the translated passages selected from generated passages
                    based on the given content
        '''
        passages = gen_file.split("\n\n")
        sentences = content.split("\n"):

        output = ""
        for i, s in enumerate(passages):
            output += self.select(s, passages[i], glove):

        return output


    def select(self, sentence, passages, glove):
        '''
        :Args:      content sentence
                    candidate passages

        :returns:   the 2-8 consecutive passages most similar to the content
                    sentence
        '''
        nlp = spacy.load('en')
        gen_doc = nlp(setence)

        gen_tokens = []
        for w in gen_doc:
            if w.pos in set(["noun","verb","adj","adv"]):
                gen_tokens.append(w.text)
                gen_tokens.extend(glove.quick_nn(w.text))

        line_scores = []
        for line in passages.split("\n"):
            content_tokens = []
            content_doc = nlp(line)

            for w in content_doc:
                if w.pos in set(["noun","verb","adj","adv"]):
                    content_tokens.append(w.text)
                    content_tokens.extend(glove.quick_nn(w.text))

            scores = [self.score(content_tokens, gen_tokens, glove) for word in line]

            line_scores.append(sum(scores)/len(scores))

        avg_scores = []
        for i in range(len(line_scores)-8):
            for l in range(2,9)
                avg_scores.append(i, l, sum(line_scores[i:i+l]/l))

        i, l = sorted(avg_scores, key=lambda x:x[2])[0]

        return "\n".join(passages.split("\n")[i:i+l])

    def score(self, tokens1, tokens2, glove):
        '''
        :Args:      a list of words (tokens)
                    a list of words (tokens)
                    a GloVe embedding object

        :returns:   the minimum similarity score between the two token lists
        '''
        s = min([min([glove.dist(t1, t2)*(1-self.pmi(set([t1, t2])) for t2 in tokens2]) for t1 in tokens1])

        return s

    def calc_pmi(self, glove):
        '''
        :Args:      GloVe embedding object from GloVe class

        :returns:   a dicitonary of tuple pmi probabilities for the content
        '''
        self.pmi = dd(lambda:0)
        c = Counter()
        words = self.content.split()
        prev = words[0]

        for w in words:
            tup = set([prev, w])
            c[tup] += 1
            prev = w

        total = sum(c.values())

        prev = words[0]

        for w in words:
            tup = set([prev, w])
            self.pmi[tup] = math.log(float(c[word])/total)
            prev = w


if __name__=="__main__":

    content_file = "data/input/nyt_article_1"
    gen_file = "data/output/gen_words"

    select = Selector(content_file, gen_file)

from collections import Counter, defaultdict as dd
from Arpabet import Arpabet
from GloVe import GloVe
import spacy
import util as u
import math, random

class Generator:

    def __init__(self, text_file, goog_file):
        self.APB = Arpabet()
        self.phonemes, self.text = self.APB.parse(text_file)
        p1 = self.to_prob(self.text.split())
        p2 = self.goog_prob(goog_file)
        self.prob = self.merge_prob(p1, p2)

    def sample(self, lstm, length=None, prime=None, temp=1):
        '''
        :Args:      an lstm to sample from
                    the length of the sample to produce (in words)
                    the primer sequence to prime the lstm
                    a temperature to sample at

        :returns:   a sample of words translated from syllables generated from
                    the lstm
        '''
        i = 0
        for line in self.phonemes:
            print sum(self.APB.nb_syllables(w) for w in line)
            i += 1
            if i == 10: break

    def to_prob(self, corpus, glove):
        '''
        :Args:      a text corpus to convert to probabilities

        :returns:   a dictionary of unigram probabilities for each word in the
                    corpus
        '''
        nlp = spacy.load('en')
        doc = nlp(corpus)

        prob = dd(lambda: 0)
        c = Counter(corpus)
        total = sum(c.values())

        for word in doc:
            count = math.log(float(c[word])/total)
            prob[word.text] = count
            if word.pos in set(["noun","verb","adj","adv"]):
                for w in glove.quick_nn[word.text]:
                    prob[w] = count

        return prob

    def goog_prob(self, data_file):
        '''
        :Args:      the data file of google's unigram counts

        :returns:   a dicitonary of unigram probabilities
        '''
        prob = dd(lambda:-float("inf"))
        c = Counter()
        with open(data_file, "r") as f:
            for line in f:
                line = line.split(", ")
                c[line[0]] += int(corpus[1])

        total = sum(c.values())
        for item, count in c.items():
            prob[item] = math.log(float(count)/total)

        return prob

    def merge_prob(self, prob1, prob2, alpha):
        '''
        :Args:      a dictionary of unigram probabilities
                    a dictionary of unigram probabilities

        :returns:   a dicitonary of unigram probabilities, merged with:
                    alpha*prob1 + (1-alpha)*prob2
        '''
        prob = dd(lambda:-float("inf"))

        for word, p1 in prob1:
            if word in prob2:
                prob[word] = alpha*p1+(1-alpha)[wo]

    def to_words(self, phonemes):
        '''
        :Args:      a list of lists of tuples of phonemes

        :returns:   the flattened string of translated words
        '''
        rv = ""
        for i, line in enumerate(phonemes):
            for phones in line:
                p = [(word, self.prob[word]) for word in self.APB.arpabet_inverse[phones]]
                if p: rv += max(p, key=lambda x:x[1])[0]+" "
            rv += "\n"
        return rv

if __name__=="__main__":

    N = 4
    T = 1000
    text_file = "data/input/eminem.txt"
    gen_file = "data/output/gen_phones.txt"

    G = Generator(text_file)

    with open(gen_file, "r") as f:
        phonemes = [[tuple(w.split()) for w in line.split("|")] for line in f]
    print G.to_words(phonemes)

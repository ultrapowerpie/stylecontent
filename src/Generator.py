from collections import Counter, defaultdict as dd
from Arpabet import Arpabet
import util as u
import math, random

class Generator:

    def __init__(self, text_file):
        self.APB = Arpabet()
        self.phonemes, self.text = self.APB.parse(text_file)


    def sample(self):
        i = 0
        for line in self.phonemes:
            print sum(self.APB.nb_syllables(w) for w in line)
            i += 1
            if i == 10: break

    def to_prob(self, corpus):
        prob = dd(lambda:-float("inf"))
        c = Counter(corpus)
        total = sum(c.values())
        for item, count in c.items():
            prob[item] = math.log(float(count)/total)

        return prob

    def to_words(self, phonemes):
        '''
        :Args:      a list of lists of tuples of phonemes

        :returns:   the flattened string of translated words
        '''
        prob = self.to_prob(self.text.split())

        rv = ""
        for i, line in enumerate(phonemes):
            for phones in line:
                p = [(word, prob[word]) for word in self.APB.arpabet_inverse[phones]]
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

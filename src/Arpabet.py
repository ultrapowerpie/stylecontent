from collections import defaultdict as dd
import re

arpabet_file = "data/static_data/arpabet.txt"
phones_file = "data/static_data/arpabet_phones.txt"

class Arpabet:

    def __init__(self):
        '''
        :Args:    a path to the arpabet dictionary file
        '''
        self.arpabet = dd(list)
        self.arpabet_inverse = dd(list)
        self.phones = dd(None)
        self.vowels = set()

        RE_D = re.compile('\d')
        with open(arpabet_file, "r") as f:
            for line in f:
                l = line.lower().split()
                if RE_D.search(l[0]): continue  #skip any words with numbers
                self.arpabet[l[0]] = tuple(RE_D.sub("", w) for w in l[1:])

        with open(phones_file, "r") as f:
            for line in f:
                l = line.lower().split()
                if l[1] == "vowel": self.vowels.add(l[0])
                self.phones[l[0]] = l[1]

        for word, syllables in self.arpabet.items():
            self.arpabet_inverse[syllables].append(word)

    def parse(self, text_file):
        '''
        :Args:    a path to the text file to parse into phonemes

        :returns: a list of lines of the text, where each line is a list of
                  words, and each word is a list of phonemes
        '''
        RE_AZ = re.compile("[^'a-z]")
        pl = []
        text = ""
        with open(text_file, "r") as f:
            for line in f:
                pl.append([])
                for word in RE_AZ.sub(" ", line.lower()).split():
                    phones = self.arpabet[word]
                    if phones: pl[-1].append(phones)
                    elif (word[-3:] == "in'"):
                        pl[-1].append(tuple(list(self.arpabet[word[:-1]+"g"][:-1])+["n"]))
                    text += word+" "
                text += "\n"
        return pl, text

    def nb_syllables(self, word):
        '''
        :Args:      a word consisting of a list of syllables

        :returns:   the number of syllables in the word
        '''
        c = 0
        for syl in word:
            if syl in self.vowels: c += 1

        return c

    def flatten(self, pl):
        '''
        :Args:      a list of lists of tuples of phonemes

        :returns:   a flattened string of lines and words for printing
        '''
        s = " | \n ".join([" | ".join([" ".join(w) for w in l]) for l in pl])
        return s+" | "

if __name__=="__main__":
    APB = Arpabet()

    # text_file = "data/input/hamilton.txt"
    text_file = "data/input/wiki1.txt"

    phonemes, text = APB.parse(text_file)

    print APB.flatten(phonemes)

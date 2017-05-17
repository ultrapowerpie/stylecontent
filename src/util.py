from collections import Counter, defaultdict as dd
import math, random, glob, os, re

def clean_file(text_file):
    with open(text_file, "r") as corpus:
        s = corpus.read()

    s = re.sub("\[(.+)\]", "", s)
    s = re.sub("\n+", "\n", s)

    with open(text_file, "w") as corpus:
        corpus.write(re.sub("\n+", "\n", s))

def compile_corpus(directory, text_file):

    s = ""
    for filename in glob.glob(directory+'*.txt'):
        clean_file(filename)
        with open(filename, "r") as f:
            s += f.read()



    with open(text_file, "w") as corpus:
        corpus.write(s)

def softmax(p, pl):
    return math.exp(p)/sum(math.exp(i) for i in pl)

def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        upto += w
        if upto >= r: return c

if __name__ == "__main__":
    directory = "data/input/eminem/"
    text_file = "data/input/eminem.txt"

    with open(text_file, "r") as f:
        print len(f.read().split())

    # clean_file(text_file)
    # compile_corpus(directory, text_file)

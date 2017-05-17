from collections import Counter, defaultdict as dd
from Arpabet import Arpabet
import util as u
import math, random

N = 4
T = 1000
text_file = "data/input/hamilton.txt"

APB = Arpabet()
phonemes, text = APB.parse(text_file)

print len(text.split())

grams = []
for line in APB.flatten(phonemes).split("\n"):
    words = line.split()
    grams += [tuple(words[i:i+N]) for i in xrange(len(words)-N+1)]

gprob = u.to_prob(grams)

gram = dd(lambda:dd(lambda:0))
gram_prob = dd(lambda:dd(lambda:0))

for key, prob in gprob.items():
    gram[key[:-1]][key[-1]] = prob

for gram1, gram1_dict in gram.items():
    for gram2, p in gram1_dict.items():
        gram_prob[gram1][gram2] = u.softmax(p, gram1_dict.values())

start = random.choice(gprob.keys())
output = " ".join(start)
last = start[1:]
for i in range(T):
    gen = u.weighted_choice(gram_prob[last].items())
    if gen: output += gen+" "
    last = last[1:]+tuple([gen])

line = output.split("|")

phonemes = []
for word in line:
    phonemes.append(tuple(word.split()))

print to_words([phonemes], text)

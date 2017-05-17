from collections import Counter, defaultdict as dd
import math, random, glob, os, re
import numpy as np
import tensorflow as tf

def clean_file(text_file):
    '''Removes all annotations and extra spaces/newlines from .txt file'''
    with open(text_file, "r") as corpus:
        s = corpus.read()

    s = re.sub("\[(.+)\]", "", s)
    s = re.sub("\n+", "\n", s)

    with open(text_file, "w") as corpus:
        corpus.write(re.sub("\n+", "\n", s))

def compile_corpus(directory, text_file):
    '''Compiles all  '''
    s = ""
    for filename in glob.glob(directory+'*.txt'):
        clean_file(filename)
        with open(filename, "r") as f:
            s += f.read()

    with open(text_file, "w") as corpus:
        corpus.write(s)

def softmax(p, pl):
    '''Returns the softmaxed probability for
    p given the probability vector pl'''
    return math.exp(p)/sum(math.exp(i) for i in pl)

def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        upto += w
        if upto >= r: return c

def run_sample(model, n, starter_syllab, primer_seq=None, temperature=1.0):
    '''Samples a length-n sequence from the model'''

    sampled_syllabs = [_ for _ in range(n)]
    current_syllab = starter_syllab
    h = np.zeros([1, model.hidden_size])
    cs = np.zeros([1, model.hidden_size])

    if primer_seq is not None:
        for c in primer_seq:
            h, cs = model.sess.run(
                [model.next_hidden, model.next_cell],
                feed_dict={
                    model.sample_input_syllab: c,
                    model.sample_input_hidden: h,
                    model.sample_cell_state: cs
                })

    for i in range(n):

        current_syllab, h, cs = model.sess.run(
            [model.next_sample, model.next_hidden, model.next_cell],
            feed_dict={
                model.sample_input_syllab: current_syllab,
                model.sample_input_hidden: h,
                model.sample_cell_state: cs,
                model.temperature: temperature})

        sampled_syllabs[i] = current_syllab

    return sampled_syllabs

def run_test(model, test_syllabs, primer_seq=None):
    '''Finds the cross entropy on a dataset.
    test_syllabs and primer_seq should be lists of ints.'''

    xentropy_accum = 0.0
    h = np.zeros([1, model.hidden_size])
    cs = np.zeros([1, model.hidden_size])

    if primer_seq is not None:
        for c in primer_seq:
            h, cs = model.sess.run(
                [model.next_hidden, model.next_cell],
                feed_dict={
                    model.sample_input_syllab: c,
                    model.sample_input_hidden: h,
                    model.sample_cell_state: cs
                })

    for i in range(len(test_syllabs) - 1):
        xentropy, h, cs  = model.sess.run(
            [model.binary_xentropy, model.next_hidden, model.next_cell],
            feed_dict={
                model.sample_input_syllab: test_syllabs[i],
                model.sample_input_hidden: h,
                model.sample_cell_state: cs,
                model.test_syllab: test_syllabs[i+1]
            })

        xentropy_accum += (xentropy / len(test_syllabs))

    xentropy_avg = xentropy_accum

    return xentropy_avg

if __name__ == "__main__":
    directory = "data/input/eminem/"
    text_file = "data/input/eminem.txt"

    with open(text_file, "r") as f:
        print len(f.read().split())

    # clean_file(text_file)
    # compile_corpus(directory, text_file)

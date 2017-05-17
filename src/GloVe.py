from collections import defaultdict as dd
from Arpabet import Arpabet
import heapq, numpy as np

class GloVe:

    def __init__(self, text_file, save_file):
        with open(text_file, "r") as f:
            for line in f:
                self.d = len(line.split())-1
                break

        self.embeddings = dd(lambda: np.zeros(self.d))
        self.APB = Arpabet()
        with open(text_file, "r") as f:
            for line in f:
                line = line.split()
                if line[0] in self.APB.arpabet:
                    self.embeddings[line[0]] = np.asarray(map(float, line[1:]))

        if save_file:
            self.save_knn(save_file)
            self.load_nn(save_file)

    def dist(self, w1, w2):
        '''
        :Args:      the first word
                    the second word

        :returns:   the Euclidean distance between the two words
        '''
        e1 = self.embeddings[w1]
        e2 = self.embeddings[w2]
        return np.linalg.norm(e1,e2)


    def knn(self, word, k=1):
        '''
        :Args:      the word to find the k nearest neighbors of
                    number of neighbors to find

        :returns:   a list of (-score, word) pairs of the k nearest neighbors
        '''
        pq = []
        glove = self.embeddings[word]

        for w, ebd in  self.embeddings.items():
            if w == word:
                continue

            if len(pq) == k:
                heapq.heappushpop(pq, (-np.linalg.norm((glove-ebd)),w))
            else:
                heapq.heappush(pq, (-np.linalg.norm((glove-ebd)),w))

        return pq

    def save_knn(self, save_file):
        '''
        :Args:      file to save the nearest neighbors data to

        :returns:   N/A
        '''
        s = ""
        for word, emb in self.embeddings:
            s += word
            s += " ".join([neighbor[1] for neighbor in self.knn(word, 5)])
            s += "\n"


        with open(save_file, "w") as f:
            f.write(s)

    def load_nn(self, save_file):
        '''
        :Args:      file to load the nearest neighbors data from

        :returns:   N/A
        '''
        self.neighbors = dd(list)
        with open(save_file, "w") as f:
            for line in f:
                line = line.split()
                self.neighbors[line[0]] = line[1:]

    def quick_nn(self, word):
        '''
        :Args:      the word to return the k nearest neighbors of

        :returns:   N/A
        '''
        return self.neighbors[word]

if __name__ == "__main__":
    text_file = "data/static_data/glove.6B/glove.6B.300d.txt"

    GV = GloVe(text_file)

    print GV.knn("ship", 100)

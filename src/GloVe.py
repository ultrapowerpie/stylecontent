from collections import defaultdict as dd
from Arpabet import Arpabet
import heapq, numpy as np

class GloVe:

    def __init__(self, text_file):
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

    def nn(self, word, k=1):
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

if __name__ == "__main__":
    text_file = "data/static_data/glove.6B/glove.6B.300d.txt"

    GV = GloVe(text_file)

    print GV.nn("ship", 100)

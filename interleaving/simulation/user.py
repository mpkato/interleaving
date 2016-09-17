import numpy as np

class User(object):
    def __init__(self, click_probs=[0, 1], stop_probs=[0, 0]):
        self.stop_probs = stop_probs
        self.click_probs = click_probs

    def examine(self, ranking, relevance):
        clicks = []
        for idx, r in enumerate(ranking):
            g = relevance.get(r, 0)
            stop_p = self.stop_probs[g]
            click_p = self.click_probs[g]
            if np.random.rand() < click_p:
                clicks.append(idx)
            if np.random.rand() < stop_p:
                break
        return clicks

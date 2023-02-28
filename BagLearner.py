import numpy as np
from scipy import stats

def author():
  return "xliu736"

class BagLearner(object):
    def __init__(self, learner, kwargs = {}, bags=10, boost = False, verbose = True):
        self.learner = learner
        self.bags = bags
        self.learners = []
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'xliu736'

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            subset = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner.add_evidence(data_x[subset], data_y[subset])

    def query(self,points):
        predicted_ys = []
        for learner in self.learners:
            predicted_ys.append(learner.query(points))
        result = stats.mode(np.array(predicted_ys))[0][0]
        return result
import numpy as np
import random
from scipy import stats

def author():
  return "xliu736"

class RTLearner(object):
    """
    This is a Random Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self,leaf_size, verbose=True):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "xliu736"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        self.rt = self.build_rt(data_x, data_y)

    def build_rt(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # build a random tree
        if data_x.shape[0] <= self.leaf_size:
            return np.array(["leaf", stats.mode(data_y)[0][0], "NA", "NA"])

        elif np.all(data_y == data_y[0]):
            return np.array(["leaf", data_y[0], "NA", "NA"])

        else:
            i = np.random.randint(0, data_x.shape[1]-1)
            splitval = np.nanmedian(data_x[:, i])
            lt_logic = data_x[:, i] <= splitval
            rt_logic = data_x[:, i] > splitval
            if len(np.unique(lt_logic)) == 1:
                return np.array(["leaf", stats.mode(data_y[lt_logic])[0][0], "NA", "NA"])
            if len(np.unique(rt_logic)) == 1:
                return np.array(["leaf", stats.mode(data_y[rt_logic])[0][0], "NA", "NA"])
            lefttree = self.build_rt(data_x[lt_logic, :], data_y[lt_logic])
            righttree = self.build_rt(data_x[rt_logic,:],data_y[rt_logic])
            if len(lefttree.shape) == 1:
                root = np.array([i, splitval, 1, 2])
            else:
                root = np.array([[i,splitval,1,lefttree.shape[0]+1]])
            random_tree = np.vstack((root,lefttree,righttree))
            return random_tree

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predicted_y = np.empty((points.shape[0],))
        #print(self.rt)
        np.set_printoptions(threshold=np.inf)
        for i in range(points.shape[0]):
            j = 0
            while self.rt[j][0] != "leaf":
                if points[i][int(float(self.rt[j][0]))] <= float(self.rt[j][1]):
                    j = j + int(float(self.rt[j][2]))
                else:
                    j = j + int(float(self.rt[j][3]))
            predicted_y[i] = self.rt[j][1]
        return predicted_y

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
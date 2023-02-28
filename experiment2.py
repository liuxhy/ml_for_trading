# Experiment 2: how changing the value of impact should affect in-sample trading behavior

import datetime as dt
from marketsimcode import compute_portvals, calculate_stat
import numpy as np
import pandas as pd
import util as ut
import matplotlib.pyplot as plt
import ManualStrategy as ms
import StrategyLearner as sl

def author():
    return "xliu736"

def exp2():
    # impact = 0, 0.005, 0.05, 0.1
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    table = np.zeros((4,3))
    plt.figure(figsize=(9, 6), dpi=300)
    impact_value = [0, 0.025, 0.05, 0.1]
    for i in range(0,4):
        j = impact_value[i]
        learner = sl.StrategyLearner(verbose=False, impact=j, commission=9.95)  # constructor
        learner.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
        learner_trades = learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase
        learner_portval = compute_portvals(learner_trades, sv, impact=j, commission=9.95)
        learner_portval = learner_portval / learner_portval[0]
        plt.plot(learner_portval, label='Impact = %s' %j)
        table[i,:] = calculate_stat(learner_portval)

    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.legend(loc='upper left')
    plt.title("Experiment 2")
    plt.savefig("images/exp2.png")
    plt.close()

    table_df = pd.DataFrame(table, index=["Impact = 0", "Impact = 0.025", "Impact = 0.05", "Impact = 0.1"],
                            columns=["Cumulative return", "Stdev of daily returns", "Mean of daily returns"])
    table_df.to_csv('p8_results.txt')

if __name__ == "__main__":
    exp2()
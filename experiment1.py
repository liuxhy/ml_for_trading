# The in-sample period is January 1, 2008 to December 31, 2009.
# The out-of-sample/testing period is January 1, 2010 to December 31, 2011.

import datetime as dt
from marketsimcode import compute_portvals
import numpy as np
import pandas as pd
import util as ut
import matplotlib.pyplot as plt
import ManualStrategy as ms
import StrategyLearner as sl

def author():
    return "xliu736"

def exp1():
    # in sample
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    prices = ut.get_data(['JPM'], pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
    prices = prices.dropna()
    benchmark_trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    benchmark_trades.iloc[0] = 1000
    benchmark_portval = compute_portvals(benchmark_trades, sv, impact=0.005, commission=9.95)

    manual_strategy = ms.ManualStrategy(verbose=False, impact=0.005, commission=9.95)  # constructor
    manual_trades, buydate, selldate = manual_strategy.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    manual_portval = compute_portvals(manual_trades, sv, impact=0.005, commission=9.95)

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    learner.add_evidence(symbol= 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)  # training phase
    learner_trades = learner.testPolicy(symbol= 'JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv=100000)  # testing phase
    learner_portval = compute_portvals(learner_trades, sv, impact=0.005, commission=9.95)

    # plotting
    benchmark_portval = benchmark_portval / benchmark_portval[0]
    manual_portval = manual_portval / manual_portval[0]
    learner_portval = learner_portval / learner_portval[0]
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(manual_portval, label="Manual Strategy", color="red")
    plt.plot(benchmark_portval, label="Benchmark", color="purple")
    plt.plot(learner_portval, label="Strategy Learner", color="green")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.legend(loc='upper left')
    plt.title("Experiment 1 - In Sample Period")
    plt.savefig("images/exp1-in.png")
    plt.close()

    # out of sample
    sv = 100000
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    prices = ut.get_data(['JPM'], pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
    prices = prices.dropna()
    benchmark_trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    benchmark_trades.iloc[0] = 1000
    benchmark_portval = compute_portvals(benchmark_trades, sv, impact=0.005, commission=9.95)

    manual_strategy = ms.ManualStrategy(verbose=False, impact=0.005, commission=9.95)  # constructor
    manual_trades, buydate, selldate = manual_strategy.testPolicy(symbol = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
    manual_portval = compute_portvals(manual_trades, sv, impact=0.005, commission=9.95)

    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    learner.add_evidence(symbol= "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)  # training phase
    learner_trades = learner.testPolicy(symbol= "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000)  # testing phase
    learner_portval = compute_portvals(learner_trades, sv, impact=0.005, commission=9.95)

    # plotting
    benchmark_portval = benchmark_portval / benchmark_portval[0]
    manual_portval = manual_portval / manual_portval[0]
    learner_portval = learner_portval / learner_portval[0]
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(manual_portval, label="Manual Strategy", color="red")
    plt.plot(benchmark_portval, label="Benchmark", color="purple")
    plt.plot(learner_portval, label="Strategy Learner", color="green")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.legend(loc='upper left')
    plt.title("Experiment 1 - Out of Sample Period")
    plt.savefig("images/exp1-out.png")
    plt.close()

if __name__ == "__main__":
    exp1()
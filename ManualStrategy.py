"""
Implementing a manual strategy

Student Name: Xinhuiyu Liu (replace with your name)
GT User ID: xliu736 (replace with your User ID)
GT ID: 903518822 (replace with your GT ID)
"""

import datetime as dt
from marketsimcode import compute_portvals, calculate_stat
import numpy as np
from indicators import *
import pandas as pd
import util as ut
import matplotlib.pyplot as plt

def author():
    return 'xliu736'

class ManualStrategy(object):
    """
    A Manual Strategy to produce a trading policy using three indicators.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float  	`
    """

    # constructor
    def __init__(self, verbose=True, impact=0.005, commission=9.95):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def testPolicy(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=10000,
    ):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[syms]
        window = 10
        # indicator 1: sma/price ratio
        ratio = get_sma(prices, window)

        # indicator 2: momentum
        momentum = get_momentum(prices, window)

        # indicator 3: bb value
        bb = get_bb(prices, window)

        trades = pd.DataFrame(0, columns=prices.columns, index=prices.index)
        share = 0
        buydate = []
        selldate = []
        for t in range(prices.shape[0]):
            if ratio.values[t, 0] < 1 and momentum.values[t-1, 0] < 0 and momentum.values[t+1, 0] > 0 or bb.values[t, 0] < -1:
                if share == 0:
                    share = 1000
                    trades.values[t, 0] = 1000
                    buydate.append(prices.index[t])
                elif share == -1000:
                    share = 1000
                    trades.values[t, 0] = 2000
                    buydate.append(prices.index[t])
            elif ratio.values[t, 0] > 1 and momentum.values[t-1,0] > 0 and momentum.values[t+1,0] < 0 or bb.values[t, 0] > 1:
                if share == 0:
                    share = -1000
                    trades.values[t, 0] = -1000
                    selldate.append(prices.index[t])
                elif share == 1000:
                    share = -1000
                    trades.values[t, 0] = -2000
                    selldate.append(prices.index[t])

        return trades, buydate, selldate

ms = ManualStrategy()
def plot_ms():
    # in sample
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    prices = ut.get_data(['JPM'], pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
    prices = prices.dropna()
    benchmark_trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    benchmark_trades.iloc[0] = 1000
    benchmark_portval = compute_portvals(benchmark_trades, sv, impact=0.005, commission=9.95)
    manual_trades, buydate, selldate = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                                                  ed=dt.datetime(2009, 12, 31), sv=100000)
    manual_portval = compute_portvals(manual_trades, sv, impact=0.005, commission=9.95)
    # plotting
    benchmark_portval = benchmark_portval / benchmark_portval[0]
    manual_portval = manual_portval / manual_portval[0]
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(manual_portval, label="Manual Strategy", color="red")
    plt.plot(benchmark_portval, label="Benchmark", color="purple")
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(buydate, ymin, ymax, color='blue', label="Long Entry Points")
    plt.vlines(selldate, ymin, ymax, color='black', label="Short Entry Points")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.legend(loc='upper left')
    plt.title("Manual Strategy - In Sample Period")
    plt.savefig("images/ms-in.png")
    plt.close()

    table = np.zeros((2, 3))
    table[0, :] = calculate_stat(benchmark_portval)
    table[1, :] = calculate_stat(manual_portval)
    table_df = pd.DataFrame(table, index=["Benchmark", "Manual Strategy"],
                            columns=["Cumulative return", "Stdev of daily returns", "Mean of daily returns"])
    table_df.to_csv('p8_ms_in.txt')

    # out of sample
    sv = 100000
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    prices = ut.get_data(['JPM'], pd.date_range(sd, ed), addSPY=False, colname="Adj Close")
    prices = prices.dropna()
    benchmark_trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    benchmark_trades.iloc[0] = 1000
    benchmark_portval = compute_portvals(benchmark_trades, sv, impact=0.005, commission=9.95)
    manual_trades, buydate, selldate = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1),
                                                                  ed=dt.datetime(2011, 12, 31), sv=100000)
    manual_portval = compute_portvals(manual_trades, sv, impact=0.005, commission=9.95)
    # plotting
    benchmark_portval = benchmark_portval / benchmark_portval[0]
    manual_portval = manual_portval / manual_portval[0]
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(manual_portval, label="Manual Strategy", color="red")
    plt.plot(benchmark_portval, label="Benchmark", color="purple")
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(buydate, ymin, ymax, color='blue', label="Long Entry Points")
    plt.vlines(selldate, ymin, ymax, color='black', label="Short Entry Points")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.legend(loc='upper left')
    plt.title("Manual Strategy - Out of Sample Period")
    plt.savefig("images/ms-out.png")
    plt.close()

    table = np.zeros((2, 3))
    table[0, :] = calculate_stat(benchmark_portval)
    table[1, :] = calculate_stat(manual_portval)
    table_df = pd.DataFrame(table, index=["Benchmark", "Manual Strategy"],
                            columns=["Cumulative return", "Stdev of daily returns", "Mean of daily returns"])
    table_df.to_csv('p8_ms_out.txt')

if __name__ == "__main__":
    plot_ms()


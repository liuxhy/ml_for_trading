import datetime as dt
import os
import numpy as np
import pandas as pd  		  	   		  	  			  		 			     			  	 
from util import get_data, plot_data

def compute_portvals(
    orders_df,
    sv,
    impact=0.005,
    commission=9.95,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    symbol = orders_df.columns
    start_date = min(orders_df.index)
    end_date = max(orders_df.index)

    # prices dataframe
    prices = get_data(symbol, pd.date_range(start_date, end_date), addSPY=False)
    prices["cash"] = 1.0
    prices = prices.dropna()

    # trades dataframe

    trades = pd.DataFrame(orders_df, index=prices.index, columns=symbol)
    trades["cash"] = 0
    trades["cash"][prices.index[0]] = 1.0 * sv
    trades["cash"] = trades["cash"].astype('float')

    for date in trades.index:
        ix = trades.index.get_loc(date)
        if trades[symbol].iloc[ix].values != 0:
            fees = prices[symbol].iloc[ix] * trades[symbol].iloc[ix] * impact + commission
            trades["cash"][ix] -= prices[symbol].iloc[ix] * trades[symbol].iloc[ix] - fees

    holdings = trades.cumsum()
    values = prices * holdings
    portvals = values.sum(axis=1)
    return portvals

def calculate_stat(portvals):
    cum_ret = (portvals[-1] / portvals[0]) - 1
    daily_ret = (portvals / portvals.shift(1)) - 1
    daily_ret.iloc[0] = 0
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std(ddof=0)

    return [cum_ret, std_daily_ret, avg_daily_ret]

def author():
    return 'xliu736'


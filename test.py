# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 21:31:21 2022

@author: xinhu
"""
import datetime as dt  		  	   		  	  			  		 			     			  	 
import random
import numpy as np
import RTLearner as rt
import BagLearner as bl
from indicators import *
import pandas as pd  		  	   		  	  			  		 			     			  	 
import util as ut  	

  	   		  	  			  		 			     			  	 
impact = 0 		  	   		  	  			  		 			     			  	 
commission = 0
learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)

symbol="IBM"  		  	   		  	  			  		 			     			  	 
sd=dt.datetime(2008, 1, 1) 		  	   		  	  			  		 			     			  	 
ed=dt.datetime(2009, 1, 1)
syms = [symbol]  		  	   		  	  			  		 			     			  	 
dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		  	  			  		 			     			  	 
prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			     			  	 
prices_SPY = prices_all["SPY"]

window = 10
# indicator 1: sma/price ratio
ratio = get_sma(prices, window)

# indicator 2: momentum
momentum = get_momentum(prices, window)

# indicator 3: bb value
bb = get_bb(prices, window)

indicators = pd.concat((ratio, momentum, bb), axis=1)
x_train = indicators.values
x_train = x_train[:-5]
        
YBUY = 0.02 + impact
YSELL = -0.02 - impact
y_train = []
for t in range(prices.shape[0] - 5):
    ret = prices[symbol].iloc[t+5] / prices[symbol].iloc[t] - 1.0
    if ret > YBUY:
        y_train.append(1)  # LONG
    elif ret < YSELL:
        y_train.append(-1)  # SHORT
    else:
        y_train.append(0)  # CASH

y_train = np.array(y_train)

""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import math  		  	   		  	  			  		 			     			  	 
import sys  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
import matplotlib.pyplot as plt
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import time
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		  	  			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  	  			  		 			     			  	 
        sys.exit(1)  		  	   		  	  			  		 			     			  	 
    inf = open(sys.argv[1])
    x = np.array([s.strip().split(",")[1:] for s in inf.readlines()[1:]])
    data = x.astype(float)

    gtid = 903518822
    np.random.seed(gtid)

    # experiments
    dt_rmse_in = np.empty((51,10)) * np.nan
    dt_rmse_out = np.empty((51,10)) * np.nan
    dt_mae_in = np.empty((51,10)) * np.nan
    dt_mae_out = np.empty((51,10)) * np.nan
    rt_mae_in = np.empty((51,10)) * np.nan
    rt_mae_out = np.empty((51,10)) * np.nan
    bg_rmse_in = np.empty((51,10)) * np.nan
    bg_rmse_out = np.empty((51,10)) * np.nan

    training_time_dt = np.empty((51, 10)) * np.nan
    training_time_rt = np.empty((51, 10)) * np.nan
    for i in range(1,51,1):
        for j in range(10):
            np.random.shuffle(data)
            # compute how much of the data is training and testing
            train_rows = int(0.6 * data.shape[0])
            # separate out training and testing data
            train_x = data[:train_rows, 0:-1]
            train_y = data[:train_rows, -1]
            test_x = data[train_rows:, 0:-1]
            test_y = data[train_rows:, -1]
            # train dt learner
            dtlearner = dt.DTLearner(leaf_size = i, verbose=True)
            start_time = time.time()
            dtlearner.add_evidence(train_x, train_y)  # train it
            training_time_dt[i,j] = time.time() - start_time
            pred_y_in = dtlearner.query(train_x)  # get the predictions
            dt_rmse_in[i,j] = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
            dt_mae_in[i, j] = np.abs(train_y - pred_y_in).sum() / train_y.shape[0]
            pred_y_out = dtlearner.query(test_x)  # get the predictions
            dt_rmse_out[i,j] = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
            dt_mae_out[i, j] = np.abs(test_y - pred_y_out).sum() / test_y.shape[0]
            # train rt learner
            rtlearner = rt.RTLearner(leaf_size=i, verbose=True)
            start_time = time.time()
            rtlearner.add_evidence(train_x, train_y)  # train it
            training_time_rt[i, j] = time.time() - start_time
            pred_y_in = rtlearner.query(train_x)  # get the predictions
            rt_mae_in[i,j] = np.abs(train_y - pred_y_in).sum() / train_y.shape[0]
            pred_y_out = rtlearner.query(test_x)  # get the predictions
            rt_mae_out[i, j] = np.abs(test_y - pred_y_out).sum() / test_y.shape[0]
            # train bag learner
            bglearner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=True)
            bglearner.add_evidence(train_x, train_y)  # train it
            pred_y_in = bglearner.query(train_x)  # get the predictions
            bg_rmse_in[i,j] = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
            pred_y_out = bglearner.query(test_x)  # get the predictions
            bg_rmse_out[i,j] = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])

    dt_rmse_in_mean = np.mean(dt_rmse_in, axis=1)
    dt_rmse_out_mean = np.mean(dt_rmse_out, axis=1)
    dt_mae_in_mean = np.mean(dt_mae_in, axis=1)
    dt_mae_out_mean = np.mean(dt_mae_out, axis=1)
    rt_mae_in_mean = np.mean(rt_mae_in, axis=1)
    rt_mae_out_mean = np.mean(rt_mae_out, axis=1)
    bg_rmse_in_mean = np.mean(bg_rmse_in, axis=1)
    bg_rmse_out_mean = np.mean(bg_rmse_out, axis=1)
    training_time_dt_mean = np.mean(training_time_dt, axis=1)
    training_time_rt_mean = np.mean(training_time_rt, axis=1)

    # plotting
    # experiment 1 plot
    plt.plot(dt_rmse_in_mean, label="In sample data")
    plt.plot(dt_rmse_out_mean, label="Out of sample data")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("Average RMSE")
    plt.title("DT Learner RMSE as a function of leaf size")
    plt.savefig('images/figure1.png')
    plt.close()

    # experiment 2 plot
    plt.plot(bg_rmse_in_mean, label="In sample data")
    plt.plot(bg_rmse_out_mean, label="Out of sample data")
    plt.title("Bag Learner RMSE as a function of leaf size")
    plt.xlabel("Leaf Size")
    plt.ylabel("Average RMSE")
    plt.legend()
    plt.savefig('images/figure2.png')
    plt.close()

    # experiment 3 plotA
    plt.plot(dt_mae_in_mean, color = 'b', label="DT In sample data")
    plt.plot(dt_mae_out_mean, color = 'b', linestyle = 'dashed', label="DT Out of sample data")
    plt.plot(rt_mae_in_mean, color = "#FFA500", label="RT In sample data")
    plt.plot(rt_mae_out_mean, color = "#FFA500", linestyle = 'dashed', label="RT Out of sample data")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("Average MAE")
    plt.title("MAE as a function of leaf size (DT vs. RT)")
    plt.savefig('images/figure3.png')
    plt.close()

    # experiment 3 plotB
    plt.plot(training_time_dt_mean, label="Decision Tree")
    plt.plot(training_time_rt_mean, label="Random Tree")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("Average time to train a single tree (s)")
    plt.title("Training time as a function of leaf size (DT vs. RT)")
    plt.savefig('images/figure4.png')
    plt.close()
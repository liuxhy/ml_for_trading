Overview
This project will introduce a strategy learner that is designed based on a
supervised machine learning algorithm and a manual strategy trader which is
manual rule-based to make decisions of trading and compare their performances
by conducting experiments.

File introduction
ManualStrategy.py: this file contains code implementing a simple manual rule-based strategy for trading decisions based on values of three indicators (price/SMA ratio, momentum and BB value).
RTLearner.py: this file contains code implementing a classification random tree learner.
BagLearner.py: this file contains code implementing a bag learner which combine the outputs of multiple instances of a single learner that is passed in
StrategyLearner.py: this file contains code implementing a strategy learner that uses a bag of RTLearner that can learn a trading policy using the same indicators used in ManualStrategy.
Indicators.py: this file contains code that calculates different indicators that can generate buy and sell signals
experiment1.py: this file contains code implementing experiement1 and generating necessary figures and tables
experiment2.py: this file contains code implementing experiement2 and generating necessary figures and tables
marketsimcode.py: this file contains code to compute portfolio values and related statistics
testproject.py: this file is a wrap of the whole project in order to generate all figures and tables in the report

Instructions on how to run the code in the ml4t environment
To produce necessary outputs for the Manual Strategy section in the report: PYTHONPATH=../:. python ManualStrategy.py
To produce necessary outputs for the Experiment 1 section in the report: PYTHONPATH=../:. python experiment1.py
To produce necessary outputs for the Experiment 2 section in the report: PYTHONPATH=../:. python experiment2.py
To produce all figures and tables in the report: PYTHONPATH=../:. python testproject.py

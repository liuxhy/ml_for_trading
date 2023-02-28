import numpy as np
from experiment1 import exp1
from experiment2 import exp2
from ManualStrategy import plot_ms

seed=1481090000
np.random.seed(seed)

def report():
    plot_ms()
    exp1()
    exp2()

def author():
  return "xliu736"

if __name__ == "__main__":
    report()

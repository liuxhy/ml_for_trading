import matplotlib.pyplot as plt
import pandas as pd

def get_sma(prices, window):
    sma = prices.rolling(window).mean()
    ratio = pd.DataFrame(sma.values/prices.values, index=prices.index, columns=prices.columns)
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    plt.grid(True)
    plot1 = ax.plot(prices, label="Daily Prices")
    plot2 = ax.plot(sma, label="20-Day Simple Moving Average")
    ax2 = ax.twinx()
    plot3 = ax2.plot(ratio, label="Ratio", color='green', linestyle = "dashed")
    ax2.set_ylabel("Ratio", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    lns = plot1 + plot2 + plot3
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc="lower left")
    plt.title("Daily Prices and 20-Day Simple Moving Average of JPM")
    plt.savefig("sma.png")
    plt.close()
    """
    return ratio

def get_momentum(prices, window):
    momentum = (prices/prices.shift(window) - 1) * 100
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    plt.grid(True)
    plot1=ax.plot(prices, label="Daily Prices", color = 'blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prices (US Dollars)", color = 'blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    plot2=ax2.plot(momentum, label="20-Day Momentum", color = 'green')
    ax2.set_ylabel("Momentum (%)", color = 'green')
    ax2.tick_params(axis='y', labelcolor='green')
    lns = plot1 + plot2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc="lower left")
    plt.title("Daily Prices and 20-Day Momentum of JPM")
    plt.savefig("momentum.png")
    plt.close()
    """
    return momentum

def get_bb(prices, window):
    rstd = prices.rolling(window).std()
    rm = prices.rolling(window).mean()
    upper_band = rm + (2 * rstd)
    lower_band = rm - (2 * rstd)
    bb = (prices - rm) / (2*rstd)
    """
    plt.figure(figsize=(10, 5), dpi=300)
    plt.subplot(211)
    plt.grid(True)
    plt.ylabel("Prices (US Dollars)")
    plt.plot(prices, label="Daily Prices")
    plt.plot(rm, label="20-Day Simple Moving Average")
    plt.plot(upper_band, label="Upper band", linestyle = "dashed")
    plt.plot(lower_band, label="Lower band", linestyle = "dashed")
    plt.legend(loc="lower left")
    plt.title("Daily Prices and 20-Day Bollinger Bands of JPM")

    plt.subplot(212)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel('BB Value')
    plt.plot(bb, label="BB Value")
    plt.legend(loc="upper right")
    plt.savefig("Bollinger Bands.png")
    plt.close()
    """
    return bb

def plot_cross(prices, window_short, window_long):
    sma_short = prices.rolling(window_short).mean()
    sma_long = prices.rolling(window_long).mean()
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(prices)
    plt.plot(sma_short)
    plt.plot(sma_long)

    for i in range(len(sma_short)-1):
        date = sma_short.index[i]
        if float(sma_short.iloc[i]) < float(sma_long.iloc[i]) and float(sma_short.iloc[i+1]) > float(sma_long.iloc[i+1]):
            plt.axvline(x=date, color = "gold", linestyle = "dashed")
        elif float(sma_short.iloc[i]) > float(sma_long.iloc[i]) and float(sma_short.iloc[i+1]) < float(sma_long.iloc[i+1]):
            plt.axvline(x=date, color = "grey", linestyle = "dashed")

    plt.xlabel("Date")
    plt.ylabel("Normalized Prices")
    plt.legend(["Daily Prices of JPM", "20-day Simple Moving Average", "100-day Simple Moving Average", "Golden Cross", "Death Cross"], loc='lower left')
    plt.title("Daily prices and Golden/Death Cross of JPM")
    plt.savefig("cross.png")
    plt.close()
    return sma_short, sma_long

def plot_macd(prices):
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line-signal_line
    plt.figure(figsize=(10, 5), dpi=300)
    plt.grid(True)
    plt.subplot(211)
    plt.grid(True)
    plt.ylabel("Prices (US Dollars)")
    plt.plot(prices, label="Daily Prices")
    plt.plot(ema_12, label="12-Day EMA")
    plt.plot(ema_26, label="26-Day EMA")
    plt.legend(loc="lower left")
    plt.title("Daily Prices and MACD Lines of JPM")
    plt.subplot(212)
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel('MACD')
    plt.plot(macd_line, label="MACD Line")
    plt.plot(signal_line, label="Signal Line")
    plt.bar(hist.index.values, hist['JPM'], width = 2, label="MACD Histogram", color="blue")
    plt.legend(loc="lower right")
    plt.savefig("macd.png")
    plt.close()
    return macd_line, signal_line

def author():
  return "xliu736"
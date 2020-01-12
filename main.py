import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from Line import TrendLine
from Graph import graph_data

# Load the stock data and calculate ema from the stock data
stock_data = yf.Ticker("^GSPC").history(period="max", rounding=True)[-25:]
print(stock_data.tail())

# # Generate trendline for given stock_data
channel_line = TrendLine(stock_data, 25)
upper = channel_line.get_upper_trendline()
lower = channel_line.get_lower_trendline()


# # Plotting something random to showcase the "tlines"
# _ = plt.figure()
# ax = plt.axes()
# x = np.linspace(0, 10, 1000)
# ax.plot(x, m_upper * x + c_upper)
# ax.plot(x, m_lower * x + c_lower)
# plt.show()
graph_data(stock_data, upper, lower, 25)

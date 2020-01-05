import matplotlib.pyplot as plt
import numpy as np

from Line import TrendLine

# Load the stock data and calculate ema from the stock data
stock_data = None
ema = None

# Generate trendline for given stock_data
channel_line = TrendLine(stock_data, ema)
m_upper, c_upper = channel_line.getUpperTrendLine()
m_lower, c_lower = channel_line.getLowerTrendLine()


# Plotting something random to showcase the "tlines"
_ = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, m_upper * x + c_upper)
ax.plot(x, m_lower * x + c_lower)
plt.show()

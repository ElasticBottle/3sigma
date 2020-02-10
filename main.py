import datetime
import json

import pandas as pd

from bull_momentum import BullMomentum
from finnhub_api import Finnhub


EXIT_NORMAL = 0
EXIT_STOP = 1


def get_data():
    datas = list()
    data = Finnhub()
    datas.append(
        data.get_data(
            "hourly",
            datetime.datetime(2017, 11, 31, 23, 30, 00),
            datetime.datetime(2019, 9, 31, 23, 30, 00),
        )
    )

    datas.append(
        data_30_minute=data.get_data(
            "30_minute",
            datetime.datetime(2017, 11, 31, 23, 30, 00),
            datetime.datetime(2019, 9, 31, 23, 30, 00),
        )
    )

    datas.append(
        data_15_minute=data.get_data(
            "15_minute",
            datetime.datetime(2017, 11, 31, 23, 30, 00),
            datetime.datetime(2019, 9, 31, 23, 30, 00),
        )
    )

    datas.append(
        data_5_minute=data.get_data(
            "5_minute",
            datetime.datetime(2017, 11, 31, 23, 30, 00),
            datetime.datetime(2018, 12, 31, 23, 30, 00),
        )
    )

    datas.append(
        data_1_minute=data.get_data(
            "1_minute",
            datetime.datetime(2017, 11, 31, 23, 30, 00),
            datetime.datetime(2018, 12, 31, 23, 30, 00),
        )
    )
    return datas


def test_datasets(datasets):
    for data in datasets:
        test_data_with_various_entry_exit(data)


def generate_entry_signals():
    # TODO: generate list of tuples for entry signals
    return []


def generate_exit_signals():
    # TODO: generate list of tuples and int for exit signal (normal and stop exit)
    return []


def test_data_with_various_entry_exit(data):
    bull_strat = BullMomentum()
    entry_timings = generate_entry_signals()
    exit_timings = generate_exit_signals()
    for entry in entry_timings:
        for exits in exit_timings:
            print(
                bull_strat.simulateStrat(
                    data, entry, exits[EXIT_NORMAL], exits[EXIT_STOP]
                )
            )


def main():
    datas = get_data()
    test_datasets(datas)


if __name__ == "__main__":
    main()


# import matplotlib.pyplot as plt
# import numpy as np
# import yfinance as yf
# from Line import TrendLine
# from Graph import graph_data

# # Load the stock data and calculate ema from the stock data
# stock_data = yf.Ticker("^GSPC").history(period="max", rounding=True)[-25:]
# print(stock_data.tail())

# # # Generate trendline for given stock_data
# channel_line = TrendLine(stock_data, 25)
# upper = channel_line.get_upper_trendline()
# lower = channel_line.get_lower_trendline()


# # # Plotting something random to showcase the "tlines"
# # _ = plt.figure()
# # ax = plt.axes()
# # x = np.linspace(0, 10, 1000)
# # ax.plot(x, m_upper * x + c_upper)
# # ax.plot(x, m_lower * x + c_lower)
# # plt.show()
# graph_data(stock_data, upper, lower, 25)

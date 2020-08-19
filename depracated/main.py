import datetime
import json

import pandas as pd

from bull_momentum import BullMomentum
from finnhub_api import Finnhub


EXIT_NORMAL = 0
EXIT_STOP = 1
DATA_INFO = 0
DATASET = 1


def get_data():
    datas = list()
    data = Finnhub()
    datas.append(
        (
            "hourly",
            data.get_crypto_data(
                "hourly",
                datetime.datetime(2017, 11, 30, 23, 30, 00),
                datetime.datetime(2019, 9, 30, 23, 30, 00),
            ),
        )
    )

    datas.append(
        (
            "30_minute",
            data.get_crypto_data(
                "30_minute",
                datetime.datetime(2017, 11, 30, 23, 30, 00),
                datetime.datetime(2019, 9, 30, 23, 30, 00),
            ),
        )
    )

    datas.append(
        (
            "15_minute",
            data.get_crypto_data(
                "15_minute",
                datetime.datetime(2017, 11, 30, 23, 30, 00),
                datetime.datetime(2019, 9, 30, 23, 30, 00),
            ),
        )
    )

    datas.append(
        (
            "5_minute",
            data.get_crypto_data(
                "5_minute",
                datetime.datetime(2017, 11, 30, 23, 30, 00),
                datetime.datetime(2018, 12, 31, 23, 30, 00),
            ),
        )
    )

    datas.append(
        (
            "1_minute",
            data.get_crypto_data(
                "1_minute",
                datetime.datetime(2017, 11, 30, 23, 30, 00),
                datetime.datetime(2018, 12, 31, 23, 30, 00),
            ),
        )
    )
    return datas


def generate_entry_signals(min_num_to_check, max_num_to_check, min_green):
    entry_signals = []
    for i in range(min_num_to_check, max_num_to_check + 1):
        for j in range(min_green, i):
            entry_signals.append((j, i))
    return entry_signals


def generate_exit_signals(
    min_exit_normal,
    max_exit_normal,
    min_exit_stop_to_check,
    max_exit_stop_to_check,
    min_red,
):
    exit_signals = []
    for i in range(min_exit_normal, max_exit_normal + 1):
        for j in range(min_exit_stop_to_check, min(max_exit_stop_to_check + 1, i + 1)):
            for k in range(min_red, j):
                exit_signals.append([i, (k, j)])
    return exit_signals


def test_datasets(datasets):
    for data in datasets:
        test_data_with_various_entry_exit(data[DATASET], data[DATA_INFO])


def test_data_with_various_entry_exit(data, data_info):
    bull_strat = BullMomentum()
    entry_timings = generate_entry_signals(3, 15, 2)
    # print(entry_timings)
    exit_timings = generate_exit_signals(5, 20, 3, 15, 2)
    # print(exit_timings)
    for entry in entry_timings:
        for exits in exit_timings:
            print(
                "simulating trade on",
                data_info,
                "\nEntry when",
                entry[0],
                "green bar in the last",
                entry[1],
                "bars",
                "\nExit normally after:",
                exits[EXIT_NORMAL],
                "bars",
                "\nStop loss exit:",
                exits[EXIT_STOP][0],
                "red bars in the last",
                exits[EXIT_STOP][1],
                "bars",
            )
            print(
                bull_strat.simulate_strat(
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

import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


def graph_data(stock_data, upper_trendline, lower_trendline, period):
    """
    Plots the data together with the trendline

    args:
        period (int): The number of days that the trendline is to be extended for.
    """
    #TODO: Fix the fact that period have to be the same as the length of the dataframe or error will be thrown. Period is basically useless right now
    date_range = pd.date_range(
        end=dt.datetime.today().replace(hour=00, minute=00, second=00, microsecond=00),
        periods=period,
    )
    date_range = date_range.to_julian_date().values.reshape(-1, 1)
    upper_predictions = upper_trendline(date_range)
    upper_trend = mpf.make_addplot(upper_predictions)
    lower_predictions = lower_trendline.predict(date_range)
    lower_trend = mpf.make_addplot(lower_predictions)
    mpf.plot(stock_data, type="candle", addplot=[upper_trend, lower_trend])

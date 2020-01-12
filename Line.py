import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression


class TrendLine:

    high = "High of Open and Close"
    low = "Low of Open and Close"

    def __init__(self, stock_data, lookback_period):
        """
        initializes the TrendLine class in the Line.py module
        
        Args:
            stock_data (pandas dataframe): Must contain 4 columns, "Open", "Close", "High", and "Low".
                Must have datatime as the index column.
                Each row correspond to a day.
            lookback_period (int): The number of days that the trendline will take into account minimally when 
                attempting to create the trendline
        """
        # ? Any missing details that might be needed to create the trendline should go here.
        self.stock_data = stock_data[-lookback_period:].copy()

    @staticmethod
    def _get_higher_dataset(input):
        """
        Helper function.
        Creates a new column in the stock_data dataframe with higher of the two values of columns "Open" and "Close" for each day.

        args:
            Input (Pandas Series): Contains values in the "Open" and "Close" fields
        """
        if input["Open"] > input["Close"]:
            return input["Open"]
        else:
            return input["Close"]

    @staticmethod
    def _get_lower_dataset(input):
        """
        Helper function.
        Creates a new column in the stock_data dataframe with lower of the two values of columns "Open" and "Close" for each day.

        args:
            Input (Pandas Series): Contains values in the "Open" and "Close" fields
        """
        if input["Open"] < input["Close"]:
            return input["Open"]
        else:
            return input["Close"]

    def get_upper_trendline(self):
        """
        Figures out the upper trendline from the stock data and returns m, c from y = m*x + c

        Returns:
            int: The gradient of the most recent resistant trendline found in stock_data
            int: The y intercept for the most recent resistant trendline found in stock_data
        """
        #TODO: Fix the documentation
        #TODO: Fix the jaggedness of trendline
        #Tried using polyfit instead of linear regression here to plot the trend line
        self.stock_data[self.high] = self.stock_data.apply(
            TrendLine._get_higher_dataset, axis=1
        ).iloc[:]
        X_train = self.stock_data.index.to_julian_date().values
        y_train = self.stock_data[self.high]
        line = np.polyfit(X_train, y_train, 1)
        return np.poly1d(line)

    def get_lower_trendline(self):
        """
        Figures out the lower trendline from the stock data and returns m, c from y = m*x + c

        Returns:
            int: The gradient of the most recent support trendline found in stock_data
            int: The y intercept for the most recent support trendline found in stock_data
        """
        #TODO: Fix the documentation
        #TODO: Fix the jaggedness of trendline
        #TODO: Fix the fact that the lower trendline appears so much higher than expected
        # Tried to use linear regression from sci-kit library instead of polyfit for the trendline
        self.stock_data[self.low] = self.stock_data.apply(
            TrendLine._get_lower_dataset, axis=1
        ).iloc[:]
        reg = LinearRegression()
        X_train = self.stock_data.index.to_julian_date().values.reshape(-1, 1)
        y_train = self.stock_data[self.low]
        print(y_train)
        reg.fit(X_train, y_train)
        return reg

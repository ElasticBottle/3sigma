class TrendLine:
    def __init__(self, stock_data, ema):
        """
        initializes the TrendLine class in the Line.py module
        
        Args:
            stock_data (pandas dataframe): Must contain 4 columns, "open", "close", "high", and "low".
                Must have datatime as the index column.
                Each row correspond to a day.
            ema (pandas dataframe): Must contain 4 columns, "open", "close", "high", and "low".
                Must have datatime as the index column.
                Each row correspond to a day.
        """
        # ? Any missing details that might be needed to create the trendline should go here.
        self.stock_data = stock_data
        self.ema = ema

    def getUpperTrendLine(self):
        """
        Figures out the upper trendline from the stock data and returns m, c from y = m*x + c

        Returns:
            int: The gradient of the most recent resistant trendline found in stock_data
            int: The y intercept for the most recent resistant trendline found in stock_data
        """
        # TODO: Implement method for getting the most recent upper trendline from the stock_data
        return 1, 4

    def getLowerTrendLine(self):
        """
        Figures out the lower trendline from the stock data and returns m, c from y = m*x + c

        Returns:
            int: The gradient of the most recent support trendline found in stock_data
            int: The y intercept for the most recent support trendline found in stock_data
        """
        # TODO: Implement method for getting the most recent lower trendline from the stock_data
        return 1, 4

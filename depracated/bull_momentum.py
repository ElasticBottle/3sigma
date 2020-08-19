import time

import pandas as pd


class TrackTrade:
    def __init__(self, open_price, close_price=-1, num_bars_since_trade_entered=0):
        super().__init__()
        print("Trade entered at:", open_price.values[0])
        self.open_price = open_price.values[0]
        self.close_price = close_price
        self.num_bars_since_trade_entered = num_bars_since_trade_entered

    def increment_num_bars_since_trade_entered(self):
        self.num_bars_since_trade_entered += 1

    def get_num_bars_since_trade_entered(self):
        return self.num_bars_since_trade_entered

    def set_close_price(self, close_price):
        self.close_price = close_price.values[0]

    def get_price_change(self):
        print(self.close_price)
        if self.close_price > -1:
            profit = self.close_price - self.open_price
            print("trade exited at:", self.close_price, "\nprofit: ", profit)
            return profit
        else:
            raise ValueError("Close price not set yet.")


class BullMomentum:
    # Costants
    NUM_BAR_TO_CHECK = 1
    NUM_GREENBAR_TO_QUALIFY_TRADE = 0
    OPEN_PRICE = "open"
    CLOSE_PRICE = "close"
    GREEN = "green"
    RED = "red"
    COMMISSION = 15.01

    # Trackers to manage trading bot
    in_trade = False
    profit = 0

    def __init__(self):
        super().__init__()

    def __enter_trade(self, timeframe_df):
        """
        Have the bot enter the trade.

        Args:
            timeframe_df (pandas dataframe): Contains the data for the bot to enter the trade on.
                Each row should be an observation in the timeline that the strat should be run on.
                Bot enters the trade based of the opening price of the most recent observation
        """
        self.in_trade = True
        self.trade = TrackTrade(timeframe_df.iloc[-1:, :][self.OPEN_PRICE])

    def __candlesticks(self, btc_usd):
        """
        Determines if the current timeframe bar is green or red

        Args:
            btc_usd (pandas series): the row that contains the current open and closing price.
                These prices are used to compute whether the bar was green(close > open) or red (cloes < open)
        """
        if btc_usd[self.CLOSE_PRICE] > btc_usd[self.OPEN_PRICE]:
            return self.GREEN
        else:
            return self.RED

    def __num_greenbars_in(self, timeframe):
        """
        Counts the number of greenbars within x amount of days/minutes

        Args:
            timeframe_df (pandas dataframe): Contains the data for the ticker to test on. 
                Each row should be an observation in the timeline that the strat should be run on.
        """
        print("in num_greenbars_in func")
        green_counter = 0
        for _, row in timeframe.iterrows():
            bar = self.__candlesticks(row)
            if bar == self.GREEN:
                green_counter += 1
        return green_counter

    def __check_for_entry(self, timeframe_df, entry):
        """
        Checks to see if conditions for entry are favourable.
        Done by looking at the past x bars in timeframe_df and looking for y green bars among them.

        Args:
            timeframe_df (pandas dataframe): Contains the data for the ticker to test on. 
                Each row should be an observation in the timeline that the strat should be run on.
            entry (tuple): first value contains the number of bars that has to be green. 
                Second value contains the number of days to look for green bars in.
                Second value should always be larger than or equal to the first value
        """
        print("In check_for_entry func")
        # if x amount of the last y bar is ‘green’, enter at market price
        if timeframe_df.shape[0] < entry[self.NUM_BAR_TO_CHECK]:
            print("not enough time has elapsed, no trades taken")
            return False
        else:
            num_green_bars = self.__num_greenbars_in(
                timeframe_df.iloc[-entry[self.NUM_BAR_TO_CHECK] :, :]
            )
            print(
                num_green_bars,
                "green bars in the last",
                entry[self.NUM_BAR_TO_CHECK],
                "bars",
            )
            if num_green_bars >= entry[self.NUM_GREENBAR_TO_QUALIFY_TRADE]:
                print("Entering trade:")
                return self.__enter_trade(timeframe_df)
            else:
                print("No trade taken")
                return False

    def __exit_trade(self, timeframe_df):
        """
        have the bot exit the trade

        Args:
            timeframe_df (pandas dataframe): Contains the data for the bot to exit the trade on.
                Each row should be an observation in the timeline that the strat should be run on.
                Bot exits the trade based of the closing price of the most recent observation
        """
        self.in_trade = False
        self.trade.set_close_price(timeframe_df.iloc[-1:, :][self.CLOSE_PRICE])
        self.profit += self.trade.get_price_change() - self.COMMISSION

    def __num_redbars_in(self, timeframe_to_check, numBars):
        return numBars - self.__num_greenbars_in(timeframe_to_check)

    def __check_stop_exit(self, timeframe_df, exit_stop):
        """
        Checks to see if conditions for exit are favourable.
        Done by looking at the past x bars in timeframe_df and looking for y red bars among them.

        Args:
            timeframe_df (pandas dataframe): Contains the data for the ticker to test on. 
                Each row should be an observation in the timeline that the strat should be run on.
            exit_stop (tuple): first value contains the number of bars that has to be red. 
                Second value contains the number of days to look for red bars in.
                Second value should always be larger than or equal to the first value
        """
        print("In check_stop_exit func")
        # if x amount of the last y bar is ‘green’, enter at market price
        if (
            self.trade.get_num_bars_since_trade_entered()
            < exit_stop[self.NUM_BAR_TO_CHECK]
        ):
            print("not enough time has elapsed, no trades taken")
            return False
        else:
            num_red_bars = self.__num_redbars_in(
                timeframe_df.iloc[-exit_stop[self.NUM_BAR_TO_CHECK] :, :],
                exit_stop[self.NUM_BAR_TO_CHECK],
            )
            print(
                num_red_bars,
                "red bars in the last",
                exit_stop[self.NUM_BAR_TO_CHECK],
                "bars",
            )
            if num_red_bars >= exit_stop[self.NUM_GREENBAR_TO_QUALIFY_TRADE]:
                print("Exiting trade:")
                return self.__exit_trade(timeframe_df)
            else:
                print("No trade taken")
                return False

    def __check_normal_exit(
        self, timeframe_df, exit_normal, num_bars_since_trade_entered
    ):
        """
        Exits the trade if the bot has been in the trade for more than exit_normal number of bars

        Args:
            timeframe_df (pandas dataframe): Contains the data for the ticker to test on. 
                Each row should be an observation in the timeline that the strat should be run on.
            exit_normal (integer): Contains the number of days to automatically exit the trade.
            num_bars_since_trade_entered (integer): Contains the numbr of days that the bot has been in the trade
        """
        if num_bars_since_trade_entered >= exit_normal:
            print("exiting normally")
            self.__exit_trade(timeframe_df)

    def __check_for_exit(
        self, timeframe_df, exit_normal, exit_stop, num_bars_since_trade_entered
    ):
        self.__check_normal_exit(
            timeframe_df, exit_normal, num_bars_since_trade_entered
        )
        self.__check_stop_exit(timeframe_df, exit_stop)

    def __decide_on_trade(self, timeframe_df, entry, exit_normal, exit_stop):
        """
        Decides whether the bot should be looking to enter or exit a trade.

        Args:
            timeframe_df (pandas dataframe): Contains the data for the ticker to test on. 
                Each row should be an observation in the timeline that the strat should be run on.
            entry (tuple): first value contains the number of bars that has to be green. 
                Second value contains the number of days to look for green bars in.
                Second value should always be larger than or equal to the first value
            exit_normal (integer): Contains the number of days to automatically exit the trade.
                Value must be larger than the second value of exit_stop.
            exit_stop (tuple): first value contains the number of bars that has to be red. 
                Second value contains the number of days to look for red bars in.
                Second value should always be larger than or equal to first value
        """
        print("In decide_on_trade func.")
        if self.in_trade:
            print("Currently in trade, looking for exit")
            self.__check_for_exit(
                timeframe_df,
                exit_normal,
                exit_stop,
                self.trade.get_num_bars_since_trade_entered(),
            )
            self.trade.increment_num_bars_since_trade_entered()
        else:
            print("Currently not in trade, looking for entry")
            self.__check_for_entry(timeframe_df, entry)

    def simulate_strat(self, timeframe_df, entry, exit_normal, exit_stop):
        """
        Simulates trading with given entry, exit_normal, and exit_stop directions

        Args:
            timeframe_df (pandas dataframe): Contains the data for the ticker to test on. 
                Each row should be an observation in the timeline that the strat should be run on.
            entry (tuple): first value contains the number of bars that has to be green. 
                Second value contains the number of days to look for green bars in.
                Second value should always be larger than or equal to the first value
            exit_normal (integer): Contains the number of days to automatically exit the trade.
                Value must be larger than the second value of exit_stop.
            exit_stop (tuple): first value contains the number of bars that has to be red. 
                Second value contains the number of days to look for red bars in.
                Second value should always be larger than or equal to first value
        """
        # ! Trading duration can be split into different segments, scored individually, and then summed to aggregate over a couple of periods

        for i in timeframe_df.index:
            print(
                "In simulate_strat func, checking on trade potential at",
                i,
                "Most recent data:\n",
                timeframe_df.loc[:i, :].tail(),
            )
            self.__decide_on_trade(
                timeframe_df.loc[:i, :], entry, exit_normal, exit_stop
            )
            time.sleep(1)
        return self.profit

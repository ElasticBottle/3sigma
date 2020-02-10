import pandas as pd
import time


class BullMomentum:

    DURATION_OF_GREENBAR = 1
    NUM_GREENBAR_TO_PASS = 0
    in_trade = False
    num_bars_since_trade_entered = 0

    def __track_trade(self):
        pass

    def __enter_trade(self):
        in_trade = True
        price_entered = 0
        # TODO fill up price entered
        self.__track_trade()

    def __candlesticks(self, btc_usd):
        for _ in btc_usd:
            if btc_usd["closing"] > btc_usd["opening"]:
                return "green"
            else:
                return "red"

    def __num_greenbars_in(self, timeframe):
        """
        Counts the number of greenbars within x amount of days/minutes
        """
        green_counter = 0
        for row in timeframe:
            bar = self.__candlesticks(row)
            if bar == "green":
                green_counter += 1
        return green_counter

    def __check_for_entry(self, timeframe_df, entry):
        # if x amount of the last y bar is ‘green’, enter at market price
        if (
            self.__num_greenbars_in(timeframe_df[-entry[self.DURATION_OF_GREENBAR]])
            >= entry[self.NUM_GREENBAR_TO_PASS]
        ):
            return self.__enter_trade()
        else:
            return False

    def __exit_trade(self):
        in_trade = True
        price_entered = 0
        # TODO track price entered
        self.__track_trade()

    def __check_stop_exit(self, exit_stop):
        pass

    def __check_normal_exit(self, exit_normal, num_bars_since_trade_entered):
        if num_bars_since_trade_entered >= exit_normal:
            self.__exit_trade()

    def __check_for_exit(self, exit_normal, exit_stop, num_bars_since_trade_entered):
        self.__check_normal_exit(exit_normal, num_bars_since_trade_entered)
        self.__check_stop_exit(exit_stop)

    def __decide_on_trade(self, timeframe_df, entry, exit_normal, exit_stop):
        if self.in_trade:
            self.__check_for_exit(
                exit_normal, exit_stop, self.num_bars_since_trade_entered
            )
            self.num_bars_since_trade_entered += 1
        else:
            self.__check_for_entry(timeframe_df, entry)

    def simulateStrat(self, timeframe_df, entry, exit_normal, exit_stop):
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
            self.__decide_on_trade(
                timeframe_df.loc[:i, :], entry, exit_normal, exit_stop
            )
            time.sleep(1)

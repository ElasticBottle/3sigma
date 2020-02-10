import pandas as pd

DURATION_OF_GREENBAR = 1
NUM_GREENBAR_TO_PASS = 0
in_trade = False


def track_trade():
    pass


def enter_trade():
    in_trade = True
    price_entered = 0
    # TODO fill up price entered
    track_trade()


def candlesticks(btc_usd):
    for _ in btc_usd:
        if btc_usd["closing"] > btc_usd["opening"]:
            return "green"
        else:
            return "red"


def num_greenbars_in(timeframe):
    """
    Counts the number of greenbars within x amount of days/minutes
    """
    green_counter = 0
    for row in timeframe:
        bar = candlesticks(row)
        if bar == "green":
            green_counter += 1
    return green_counter


def check_for_entry(timeframe_df, entry):
    # if x amount of the last y bar is ‘green’, enter at market price
    if (
        num_greenbars_in(timeframe_df[-entry[DURATION_OF_GREENBAR]])
        >= entry[NUM_GREENBAR_TO_PASS]
    ):
        return enter_trade()
    else:
        return False


def exit_trade():
    in_trade = True
    price_entered = 0
    # TODO track price entered
    track_trade()


def check_stop_exit(exit_stop):
    pass


def check_normal_exit(exit_normal, num_bars_since_trade_entered):
    if num_bars_since_trade_entered >= exit_normal:
        exit_trade()


def check_for_exit(exit_normal, exit_stop, num_bars_since_trade_entered):
    check_normal_exit(exit_normal, num_bars_since_trade_entered)
    check_stop_exit(exit_stop)


def decide_on_trade(timeframe_df, entry, exit_normal, exit_stop):
    if in_trade:
        check_for_exit(exit_normal, exit_stop, num_bars_since_trade_entered)
        num_bars_since_trade_entered += 1
    else:
        check_for_entry(timeframe_df, entry)


def simulateStrat(timeframe_df, entry, exit_normal, exit_stop):
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
    # Trading duration should be split into different segments, scored individually, and then summed to aggregate over a couple of periods

    # for i in number of values in dataframe:
    decide_on_trade(timeframe_df.iloc[0:i, :], entry, exit_normal, exit_stop)
